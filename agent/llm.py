"""
LLM layer -- the only place that talks to the model.

Uses Google's Gemini API (free tier, no card required) via the
google-genai SDK. Everything else in the system (safety layer,
taxonomy/routing, orchestrator, Streamlit UI) is provider-agnostic
and unaffected by this choice -- this file is the only integration
point, by design, so swapping providers later only means touching
this one file.

Design decisions:

1. TWO calls per turn, not one. Gemini would often return ONLY a
   function call with no text when asked to do both at once (unlike
   Claude, which reliably mixes text + tool_use). So this is split:
     - get_reply()     -- plain conversational turn, no tools, always
                           produces real text. Uses REPLY_MODEL.
     - extract_state() -- a second call with the tool call FORCED
                           (tool_config mode="ANY"). Uses the lighter
                           EXTRACTION_MODEL, since structured
                           classification doesn't need the heavier model.

2. These two calls run SEQUENTIALLY, not in parallel. A parallel
   version was tried and reverted: extraction was built from history
   that already ended on a user turn, and Gemini strictly rejects two
   consecutive turns from the same role -- this caused hard failures
   on every turn, not occasional slowness. Sequential costs a bit more
   latency but is the version that's actually reliable, which matters
   far more for a live demo than shaving off a second or two.

3. Once a recommendation exists (routed_category is set on the
   state), get_reply is given the actual service catalog as grounding
   context, so if the user pushes back on the suggestion the model
   can't improvise alternatives that don't exist in the real taxonomy.
"""
import os
import time
from typing import Tuple, Dict, Any

from google import genai
from google.genai import types, errors

# Extraction is simple structured classification -- doesn't need the
# heavier model, so it uses a lighter/faster one even though calls are
# sequential.
REPLY_MODEL = "gemini-3.6-flash"
EXTRACTION_MODEL = "gemini-3.5-flash-lite"

REPLY_SYSTEM_PROMPT = """You are a warm, non-judgmental intake assistant for a mental \
health platform. Your job is NOT to diagnose or provide therapy -- it is to understand, \
in a few gentle turns, what kind of support someone might need.

Guidelines:
- Ask at most one question per turn. Keep replies short and human, not clinical.
- Do not diagnose, do not suggest medication, do not offer therapy yourself.
- Do not decide you have enough information after just 1-2 exchanges, even for symptoms \
that seem clear-cut -- a couple of short answers is rarely enough context. Aim for at \
least 3 back-and-forth exchanges before signaling you're ready to suggest a next step.
- Once you do have a real sense of the topic, how long it's been going on, and how much \
it's affecting the person, let them know you have a suggestion rather than asking more \
questions.
"""

EXTRACTION_SYSTEM_PROMPT = """You are analyzing a support-intake conversation so far. \
Call update_conversation_state with your best-effort read of the situation. Always call \
the function -- never skip it."""

UPDATE_STATE_DECLARATION = types.FunctionDeclaration(
    name="update_conversation_state",
    description="Record the current understanding of the user's situation after this turn.",
    parameters={
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "Short label for what the conversation is about, e.g. 'work stress', 'grief', 'relationship conflict'.",
            },
            "severity": {
                "type": "string",
                "enum": ["low", "moderate", "high"],
                "description": "Best-effort read of how serious/impairing this seems, based on what the user has shared.",
            },
            "ready_to_route": {
                "type": "boolean",
                "description": "True once enough is understood to suggest a next step (usually after 3+ turns).",
            },
        },
        "required": ["topic", "severity", "ready_to_route"],
    },
)

TOOLS = [types.Tool(function_declarations=[UPDATE_STATE_DECLARATION])]

FORCE_TOOL_CONFIG = types.ToolConfig(
    function_calling_config=types.FunctionCallingConfig(
        mode="ANY",
        allowed_function_names=["update_conversation_state"],
    )
)


def _client() -> genai.Client:
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def _generate_with_retry(client: genai.Client, max_attempts: int = 2, delay_seconds: float = 1.5, **kwargs):
    """
    Wraps client.models.generate_content with a short retry for transient
    server-side overload (503 ServerError). ClientError (4xx, e.g. bad
    request, invalid key) is NOT retried, since retrying won't fix those.
    """
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return client.models.generate_content(**kwargs)
        except errors.ServerError as e:
            last_exc = e
            if attempt < max_attempts - 1:
                time.sleep(delay_seconds * (attempt + 1))
    raise last_exc


def _to_gemini_contents(history: list) -> list:
    """Converts our simple [{"role": "user"/"assistant", "content": str}] history
    into Gemini's Content objects (Gemini uses 'model' instead of 'assistant')."""
    contents = []
    for msg in history:
        role = "model" if msg["role"] == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
    return contents


def _service_catalog_text() -> str:
    """Renders the real taxonomy as grounding text for the reply model,
    so it can't invent services that don't exist in agent/taxonomy.py."""
    from .taxonomy import SERVICES
    lines = [f"- {s['name']}: {s['description']}" for s in SERVICES]
    return "\n".join(lines)


def get_reply(history: list, already_routed: bool = False) -> str:
    """Plain conversational turn -- no tools, guaranteed to return text.

    If already_routed is True, the system prompt is extended with the
    real service catalog so any follow-up discussion of alternatives
    stays grounded in services that actually exist.
    """
    client = _client()
    system_prompt = REPLY_SYSTEM_PROMPT
    if already_routed:
        system_prompt += (
            "\n\nYou have already suggested a next step to this person. If they push back "
            "or ask about alternatives, ONLY reference these real services -- never invent "
            "others:\n" + _service_catalog_text()
        )
    response = _generate_with_retry(
        client,
        model=REPLY_MODEL,
        contents=_to_gemini_contents(history),
        config=types.GenerateContentConfig(system_instruction=system_prompt),
    )
    text = ""
    for part in response.candidates[0].content.parts:
        if part.text:
            text += part.text
    return text.strip()


def extract_state(history_including_reply: list) -> Dict[str, Any]:
    """Second call, with the tool call forced, so extraction always runs.

    Note: Gemini's generate_content rejects any request whose last turn is
    from the model. So instead of ending on the assistant's reply, we
    append one more explicit user-role instruction asking for the
    extraction -- keeping the conversation ending on a user turn, as the
    API requires.
    """
    client = _client()
    contents = _to_gemini_contents(history_including_reply)
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(
                text="Based on the conversation so far, call update_conversation_state now."
            )],
        )
    )
    response = _generate_with_retry(
        client,
        model=EXTRACTION_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=EXTRACTION_SYSTEM_PROMPT,
            tools=TOOLS,
            tool_config=FORCE_TOOL_CONFIG,
        ),
    )
    for part in response.candidates[0].content.parts:
        if part.function_call and part.function_call.name == "update_conversation_state":
            return dict(part.function_call.args)
    return {}


def get_reply_and_state(history: list, already_routed: bool = False) -> Tuple[str, Dict[str, Any]]:
    """
    Calls Gemini twice, sequentially: once for the conversational reply,
    once (forced) for structured state extraction. Returns
    (assistant_reply_text, extracted_state).
    Requires GEMINI_API_KEY to be set in the environment.
    """
    reply_text = get_reply(history, already_routed=already_routed)
    updated_history = history + [{"role": "assistant", "content": reply_text}]
    extracted_state = extract_state(updated_history)
    return reply_text, extracted_state
