import argparse

try:
    from .app import run_anthropic_mission
except ImportError:
    from app import run_anthropic_mission


def demo(mission: str) -> None:
    out = run_anthropic_mission(mission)
    print("[Anthropic] primary:", out.get("primary"))
    print("[Anthropic] support:", out.get("support"))
    print("[Anthropic] result:", out.get("result"))
    print("[Anthropic] verification:", out.get("verification"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", default="secure llm tool use with mcp")
    args = parser.parse_args()
    demo(args.mission)
