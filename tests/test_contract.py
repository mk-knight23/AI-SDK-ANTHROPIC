from app import run_anthropic_mission


def test_mission_contract_returns_routing_and_verification():
    result = run_anthropic_mission("build secure api, add tests, and deploy")

    assert result["primary"]
    assert isinstance(result["support"], list)
    assert result["verification"]


def test_skill_routing_selects_security_for_audit_mission():
    result = run_anthropic_mission("secure audit and threat model the workflow")

    assert result["primary"] == "SENTINEL"
