"""
Runs each scripted conversation in conversations.py through the real
orchestrator (requires GEMINI_API_KEY to be set), saves the full
transcript + recommendation to evals/results/<name>.md, and prints a
pass/fail summary based on simple assertions.

This is checked into the repo so reviewers can see actual agent
behavior on a spread of scenarios, not just a happy-path demo.

Usage:
    export GEMINI_API_KEY=...
    python evals/run_evals.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.state import ConversationState
from agent.orchestrator import process_turn
from conversations import CONVERSATIONS

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_one(convo: dict) -> dict:
    state = ConversationState()
    transcript_lines = []
    saw_crisis = False
    final_recommendation = None

    for turn in convo["turns"]:
        state, reply, recommendation = process_turn(state, turn)
        transcript_lines.append(f"**User:** {turn}\n")
        transcript_lines.append(f"**Assistant:** {reply}\n")
        if recommendation:
            final_recommendation = recommendation
            transcript_lines.append(
                f"**Recommendation:** {recommendation['category']} — {recommendation['reasoning']}\n"
            )
        if state.crisis_flag:
            saw_crisis = True

    passed = saw_crisis == convo["expect_crisis"]

    return {
        "name": convo["name"],
        "passed": passed,
        "expect_crisis": convo["expect_crisis"],
        "saw_crisis": saw_crisis,
        "final_state": state.as_dict(),
        "transcript_md": "\n".join(transcript_lines),
    }


def main():
    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY not set -- cannot run live evals. See README for details.")
        sys.exit(1)

    RESULTS_DIR.mkdir(exist_ok=True)
    results = []

    for convo in CONVERSATIONS:
        print(f"Running: {convo['name']}...")
        result = run_one(convo)
        results.append(result)

        out_path = RESULTS_DIR / f"{convo['name']}.md"
        out_path.write_text(
            f"# {convo['name']}\n\n"
            f"Expected crisis flag: {result['expect_crisis']} | "
            f"Actual: {result['saw_crisis']} | "
            f"**{'PASS' if result['passed'] else 'FAIL'}**\n\n"
            f"Final state: `{result['final_state']}`\n\n"
            f"---\n\n{result['transcript_md']}"
        )

    print("\nSummary:")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['name']}")

    n_failed = sum(1 for r in results if not r["passed"])
    if n_failed:
        print(f"\n{n_failed} scenario(s) failed.")
        sys.exit(1)
    else:
        print("\nAll scenarios passed.")


if __name__ == "__main__":
    main()
