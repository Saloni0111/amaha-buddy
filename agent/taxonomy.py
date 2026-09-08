"""
Routing / lightweight RAG layer.

Design decision: the final recommendation should be grounded in a
real service catalog, not freely generated text -- so this module
owns the mapping from "what the conversation is about" to "which
concrete service fits", separately from the LLM's conversational
reply.

For this demo, matching is done with simple keyword overlap against
each service's description. In production this would be replaced
with real embeddings over Amaha's actual service/content catalog and
a proper vector store (see README "future improvements").
"""
from typing import Dict, List, Tuple

SERVICES: List[Dict[str, str]] = [
    {
        "name": "Self-help content",
        "description": (
            "Curated articles, exercises and guided content for mild, "
            "everyday stress, low mood, or wanting to build healthier habits. "
            "No appointment needed."
        ),
        "keywords": "stress anxious overwhelmed busy tired habit sleep motivation mild everyday",
    },
    {
        "name": "Coaching",
        "description": (
            "Structured sessions with a coach for goal-setting, career "
            "transitions, habit change, or building specific skills like "
            "communication or confidence."
        ),
        "keywords": "career goals confidence habit change productivity skills coaching direction",
    },
    {
        "name": "Therapy",
        "description": (
            "Sessions with a licensed therapist for ongoing emotional "
            "difficulty, anxiety, depression, grief, relationship issues, "
            "or patterns that have been going on for a while."
        ),
        "keywords": "anxiety depression grief relationship breakup sad lonely trauma ongoing pattern persistent",
    },
    {
        "name": "Psychiatry",
        "description": (
            "Consultation with a psychiatrist for evaluation of symptoms "
            "that may benefit from medical/medication support, or "
            "significant, severe or long-standing symptoms."
        ),
        "keywords": "severe medication diagnosis panic attacks insomnia appetite concentration severe symptoms",
    },
    {
        "name": "Crisis support",
        "description": (
            "Immediate crisis resources and helplines for anyone in "
            "danger or in acute distress right now."
        ),
        "keywords": "crisis emergency danger suicide self harm immediate urgent",
    },
]


def _stem(word: str) -> str:
    """Very light stemming so 'depressed'/'depression', 'anxious'/'anxiety' etc.
    have a chance of overlapping. Not linguistically rigorous -- good enough
    for a demo-scale keyword match. Production should use real embeddings."""
    word = word.strip(".,!?").lower()
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) > 3:
            return word[: -len(suffix)]
    return word


def _score(text: str, keywords: str) -> int:
    text_stems = {_stem(w) for w in text.lower().split()}
    kw_stems = {_stem(w) for w in keywords.lower().split()}
    # also give partial credit for stem-prefix overlap (e.g. "anxious" vs "anxiety")
    exact = len(text_stems & kw_stems)
    prefix_bonus = sum(
        1 for tw in text_stems for kw in kw_stems
        if tw != kw and len(tw) > 4 and len(kw) > 4 and (tw[:5] == kw[:5])
    )
    return exact + prefix_bonus


# Severity extracted by the LLM (low/moderate/high) biases routing toward
# services meant for that intensity, so a keyword hit on physical symptoms
# doesn't override an explicitly higher severity read -- keyword overlap
# alone was routing "moderate" severity into "mild, everyday" self-help.
SEVERITY_BIAS: Dict[str, Dict[str, int]] = {
    "low": {"Self-help content": 3, "Coaching": 1},
    "moderate": {"Coaching": 2, "Therapy": 3, "Self-help content": 1},
    "high": {"Therapy": 3, "Psychiatry": 3},
}


def route(conversation_text: str, severity: str = None) -> Tuple[Dict[str, str], List[Tuple[str, int]]]:
    """
    Matches the conversation so far against the service taxonomy.
    `severity` (low/moderate/high, from the LLM's extraction) biases the
    result toward services suited to that intensity -- see SEVERITY_BIAS.
    Returns the best-matching service plus all scores (for eval/debug visibility).
    """
    bias = SEVERITY_BIAS.get(severity, {})
    scores = [
        (s["name"], _score(conversation_text, s["keywords"]) + bias.get(s["name"], 0))
        for s in SERVICES
    ]
    scores.sort(key=lambda x: x[1], reverse=True)
    best_name = scores[0][0]
    best_service = next(s for s in SERVICES if s["name"] == best_name)
    return best_service, scores
