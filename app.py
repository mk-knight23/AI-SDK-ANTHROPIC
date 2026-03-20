"""Production-style Anthropic runtime for Kazi's Agents Army."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent / "core"))
from agents_army_core import MissionRequest, build_mission_plan, render_system_instructions


def run_anthropic_mission(mission_text: str) -> dict:
    plan = build_mission_plan(MissionRequest(mission_text))
    instructions = render_system_instructions(plan)

    try:
        from anthropic import Anthropic
    except Exception as exc:
        return {
            "primary": plan.primary,
            "support": plan.support,
            "result": None,
            "verification": f"Anthropic dependency missing: {exc}",
        }

    _ = Anthropic()
    return {
        "primary": plan.primary,
        "support": plan.support,
        "result": instructions,
        "verification": "Anthropic client initialization succeeded.",
    }
