"""
Safety layer.

Design decision: this runs on every single user message, BEFORE the
main orchestrator/LLM logic touches it, and is completely independent
of conversational state. The reasoning: trusting a single
conversational LLM call to "notice" risk and decide to escalate is
fragile -- the model might be mid-flow asking something unrelated.
A dedicated, always-on check removes that dependency.

This is a lightweight keyword-based classifier for demo purposes.
In production this should be replaced with a more robust, clinically
reviewed classifier (and ideally a secondary model pass for
ambiguous phrasing), and tuned in partnership with Amaha's clinical
team rather than by an engineer's best guess at keywords.

Bias intentionally toward over-triggering: a false positive costs the
user a moment of "I'm fine, keep going", a false negative could cost
much more.
"""
import re

# Pattern-level signals, not exhaustive -- intentionally kept broad
# rather than an exhaustive/mechanism-level list.
_CRISIS_PATTERNS = [
    r"\bkill (myself|me)\b",
    r"\bend(ing)? (my|it all)\b.*\blife\b",
    r"\bsuicid",
    r"\bwant to die\b",
    r"\bdon'?t want to (be alive|live)\b",
    r"\bno reason to (live|go on)\b",
    r"\bcan'?t go on\b",
    r"\bhurt(ing)? myself\b",
    r"\bself[\s-]?harm",
    r"\bbetter off (dead|without me)\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _CRISIS_PATTERNS]


def check_crisis(user_message: str) -> bool:
    """Returns True if the message contains a crisis-level risk signal."""
    if not user_message:
        return False
    return any(p.search(user_message) for p in _COMPILED)


CRISIS_RESPONSE = (
    "I'm really glad you told me this, and I want to make sure you get "
    "support right now, not just a chat. If you're in immediate danger, "
    "please contact local emergency services.\n\n"
    "You can also reach:\n"
    "- iCall (India): 9152987821\n"
    "- Vandrevala Foundation: 1860-2662-345\n"
    "- AASRA: +91-9820466726\n\n"
    "Would it help to talk about what's going on while you consider reaching out to one of these?"
)
