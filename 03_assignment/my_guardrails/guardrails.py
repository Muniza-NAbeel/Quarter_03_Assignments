from agents import input_guardrail, GuardrailFunctionOutput

@input_guardrail # type: ignore
def check_input(ctx, agent, user_input: str) -> GuardrailFunctionOutput:
    """Block negative or sensitive input."""
    text = user_input.lower()
    negative_terms = ["idiot", "stupid", "hate", "pathetic", "refund", "return"]
    if any(term in text for term in negative_terms):
        # Tripwire triggered: stop the bot agent
        return GuardrailFunctionOutput(
            output_info="Inappropriate or complex query detected",
            tripwire_triggered=True
        )
    return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)