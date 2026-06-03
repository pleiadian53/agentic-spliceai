import openai
from aisuite import Client

from nexus.llm.parsing import is_responses_model, normalize_model_name


def call_llm_text(client: Client, model: str, messages: list[dict[str, str]], temperature: float = 1.0) -> str:
    """
    Call LLM for text generation, supporting both Chat and Responses APIs.
    """
    aisuite_model, openai_model = normalize_model_name(model)

    # === Path B: Responses API ===
    if is_responses_model(openai_model):
        native_client = openai.OpenAI()
        try:
            # Use input list pattern (Canonical)
            # Note: Responses API handles list of messages as conversation history
            response = native_client.responses.create(
                model=openai_model,
                input=messages,
                temperature=temperature
            )
            
            # Extract text content
            texts = []
            if hasattr(response, 'output'):
                for item in response.output:
                    if hasattr(item, 'content'):
                        if isinstance(item.content, str): texts.append(item.content)
                        elif hasattr(item.content, 'text'): texts.append(item.content.text)
            
            # Fallback if no content in output
            if not texts and hasattr(response, 'output_text'):
                return response.output_text
                
            return "\n".join(texts)
            
        except Exception as e:
            return f"[LLM Error: {e}]"

    # === Path A: Chat API (aisuite) ===
    else:
        try:
            response = client.chat.completions.create(
                model=aisuite_model,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[LLM Error: {e}]"


# `call_llm_with_tools` moved to nexus.llm.tool_loop (2026-06-02 refactor).
# Import from there directly:
#     from nexus.llm.tool_loop import call_llm_with_tools
