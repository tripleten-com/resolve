"""billing_agent: the shared loop with the billing prompt, plus get_account.

The one rule that lives in code, not in the prompt: the email passed to get_account must equal
the ticket's own customer_email, or the call is refused before it reaches the server.
"""
from __future__ import annotations

from resolve.graph.agents import run_agent
from resolve.graph.prompts import BILLING_AGENT


def check_billing_call(name: str, args: dict, state: dict) -> str | None:
    if name != "get_account":
        return None
    own = (state.get("ticket", {}).get("customer_email") or "").strip().lower()
    asked = (args.get("email") or "").strip().lower()
    if asked != own:
        return (f"refused: get_account may only be called for the ticket's own customer ({own}); "
                f"{asked or '(empty)'} is another customer. Do not look it up; hand off to a human instead.")
    return None


def billing_agent(state: dict) -> dict:
    return run_agent("billing_agent", BILLING_AGENT, state, call_check=check_billing_call)
