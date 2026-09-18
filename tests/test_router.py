from resolve.graph.nodes.route import decide
from resolve.graph.nodes.router import after_classify, after_input_guard, after_review, after_route
from resolve.graph.state import initial_state


def good_state(**over):
    s = initial_state("T-1", {"subject": "s", "body": "b", "customer_email": "a@b.c"})
    s["classification"] = {"type": "how_to", "urgency": "normal", "aimed_at_agent": False, "confidence": 0.9}
    s["review"] = {"verdict": "approve", "reasons": [], "revision_count": 0}
    s["confidence"] = 0.9
    s["guard_output"] = {"flags": [], "ok": True}
    s.update(over)
    return s


def test_input_guard_edge():
    assert after_input_guard({"guard_input": {"quarantined": True}}) == "escalate"
    assert after_input_guard({"guard_input": {"quarantined": False}}) == "classify"


def test_classify_edge_routes_each_type_and_escalates_security_or_aimed():
    for t, node in [("how_to", "knowledge_agent"), ("billing", "billing_agent"), ("bug", "bug_agent"), ("security", "escalate")]:
        assert after_classify({"classification": {"type": t, "aimed_at_agent": False}}) == node
    assert after_classify({"classification": {"type": "how_to", "aimed_at_agent": True}}) == "escalate"
    assert after_classify({"classification": {"type": "nonsense", "aimed_at_agent": False}}) == "escalate"


def test_review_edge_allows_exactly_one_revision():
    s = {"classification": {"type": "billing"}}
    assert after_review({**s, "review": {"verdict": "approve", "revision_count": 0}}) == "output_guard"
    assert after_review({**s, "review": {"verdict": "revise", "revision_count": 1}}) == "billing_agent"
    assert after_review({**s, "review": {"verdict": "revise", "revision_count": 2}}) == "output_guard"


def test_route_auto_resolves_only_when_everything_holds():
    assert decide(good_state()) == ("auto_resolve", "")
    assert decide(good_state(confidence=0.69))[0] == "escalate"
    assert decide(good_state(confidence=0.70))[0] == "auto_resolve"
    assert decide(good_state(review={"verdict": "revise", "reasons": ["x"], "revision_count": 2}))[0] == "escalate"
    assert decide(good_state(guard_output={"flags": ["link_stripped:evil.example"], "ok": False}))[0] == "escalate"
    bug = good_state()
    bug["classification"]["type"] = "bug"
    assert decide(bug) == ("escalate", "type bug never auto-resolves")
    billing = good_state()
    billing["classification"]["type"] = "billing"
    assert decide(billing)[0] == "auto_resolve"


def test_refunds_and_duplicate_charges_always_escalate():
    s = good_state()
    s["classification"]["type"] = "billing"
    s["ticket"] = {"subject": "Refund request for Aug 25 charge", "body": "please refund"}
    assert decide(s) == ("escalate", "refund or duplicate charge: the billing team decides")
    s["ticket"] = {"subject": "Question about a charge", "body": "We were charged twice on the 14th."}
    assert decide(s)[0] == "escalate"
    s["ticket"] = {"subject": "Cancel subscription", "body": "Will I get a refund for unused days?"}
    assert decide(s)[0] == "auto_resolve"  # refund only in the body of a cancellation question
    s["classification"]["type"] = "how_to"
    s["ticket"] = {"subject": "Refund request", "body": "x"}
    assert decide(s)[0] == "auto_resolve"  # the rule is for billing tickets only


def test_route_edge_reads_decision():
    assert after_route({"decision": "auto_resolve"}) == "resolve"
    assert after_route({"decision": "escalate"}) == "escalate"
    assert after_route({}) == "escalate"
