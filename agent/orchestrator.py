"""
Orchestrator -- the spine of the agent.

Per-turn flow:
  1. Safety check (always runs, independent of everything else).
     If triggered: return crisis resources immediately, skip the rest.
  2. Otherwise, call the LLM (with tool-calling) to get a reply and an
     updated structured state.
  3. If the state says we're ready to route, call the taxonomy/RAG
     module to produce a grounded recommendation with reasoning.
"""
from typing import Tuple, Optional, Dict

from .state import ConversationState
from .safety import check_crisis, CRISIS_RESPONSE
from .llm import get_reply_and_state
from .taxonomy import route

# Don't let the LLM route after just 1-2 short exchanges, even if it says
# it's ready -- the model has been observed to declare ready_to_route
# prematurely on brief symptom mentions alone. This is a deterministic
# floor the orchestrator enforces regardless of what the model claims,
# so behavior stays testable rather than fully at the model's discretion.
MIN_TURNS_BEFORE_ROUTE = 3


def process_turn(state: ConversationState, user_message: str) -> Tuple[ConversationState, str, Optional[Dict]]:
    """
    Advances the conversation by one turn.
    Returns (updated_state, assistant_reply_text, recommendation_or_None).
    """
    state.add_message("user", user_message)
    state.turn_count += 1

    # 1. Safety check -- always on, always first.
    if check_crisis(user_message):
        state.crisis_flag = True
        state.add_message("assistant", CRISIS_RESPONSE)
        return state, CRISIS_RESPONSE, {
            "category": "Crisis support",
            "reasoning": "A crisis-level risk signal was detected in your message, so support resources are shown immediately.",
        }

    # 2. Main conversational turn via the LLM, with tool-calling for state extraction.
    # If a recommendation has already been made this session, pass that
    # through so the reply stays grounded in the real service catalog
    # instead of improvising alternatives that don't exist (see llm.py).
    already_routed = state.routed_category is not None

    # If the API is unavailable (e.g. transient overload not solved by the
    # retry inside llm.py), fail gracefully rather than crashing the app --
    # a raw traceback is a bad experience for a mental-health-adjacent tool.
    try:
        reply_text, extracted = get_reply_and_state(state.history, already_routed=already_routed)
    except Exception as e:
        # Print the real error to the terminal (not shown to the user) --
        # a generic "temporary hiccup" message is fine for the chat UI, but
        # swallowing the actual exception makes real bugs hard to catch.
        print(f"[AmahaBuddy] LLM call failed: {type(e).__name__}: {e}", flush=True)
        fallback_text = (
            "I'm having trouble connecting right now. This isn't about what you shared -- "
            "just a temporary hiccup on my end. Could you try sending that again in a moment?"
        )
        state.add_message("assistant", fallback_text)
        return state, fallback_text, None

    state.add_message("assistant", reply_text)

    if extracted:
        state.topic = extracted.get("topic", state.topic)
        state.severity = extracted.get("severity", state.severity)
        state.ready_to_route = extracted.get("ready_to_route", state.ready_to_route)

    # 3. Routing, once the LLM signals we have enough context AND we've had
    # a minimum number of exchanges (see MIN_TURNS_BEFORE_ROUTE above).
    recommendation = None
    if state.ready_to_route and not state.routed_category and state.turn_count >= MIN_TURNS_BEFORE_ROUTE:
        conversation_text = " ".join(m["content"] for m in state.history if m["role"] == "user")
        best_service, scores = route(conversation_text, severity=state.severity)
        state.routed_category = best_service["name"]
        recommendation = {
            "category": best_service["name"],
            "description": best_service["description"],
            "reasoning": (
                f"Based on what you've shared (topic: {state.topic}, severity: {state.severity}), "
                f"{best_service['name'].lower()} looks like the best fit."
            ),
            "all_scores": scores,
        }

    return state, reply_text, recommendation
