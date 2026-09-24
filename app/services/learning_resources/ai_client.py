import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Gemini 2.5 models "think" before answering by default, and that thinking is
# billed against max_tokens. On short/ambiguous note content the model can
# burn the whole budget thinking, cutting off the actual JSON reply mid-string
# and breaking json.loads() downstream — disable thinking for this small
# extraction task rather than guessing a larger token budget.
GEMINI_NO_THINKING_KWARGS = {
    "extra_body": {"extra_body": {"google": {"thinking_config": {"thinking_budget": 0}}}}
}


def _build_client() -> tuple[AsyncOpenAI, str, str]:
    """Build the LLM client based on AI_PROVIDER (or whichever API key is available).

    Gemini is accessed through Google's OpenAI-compatible endpoint, so callers
    that use chat.completions.create with json_schema response_format work
    unchanged regardless of provider.
    """
    provider = os.getenv("AI_PROVIDER", "").strip().lower()

    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if not provider:
        provider = "openai" if openai_key else "gemini"

    if provider == "gemini":
        if not gemini_key:
            raise RuntimeError("AI_PROVIDER is 'gemini' but GEMINI_API_KEY is not set.")
        return (
            AsyncOpenAI(api_key=gemini_key, base_url=GEMINI_OPENAI_BASE_URL),
            os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            provider,
        )

    if provider == "openai":
        if not openai_key:
            raise RuntimeError("AI_PROVIDER is 'openai' but OPENAI_API_KEY is not set.")
        return AsyncOpenAI(api_key=openai_key), os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), provider

    raise RuntimeError(f"Unknown AI_PROVIDER '{provider}'. Use 'openai' or 'gemini'.")


client, MODEL_NAME, AI_PROVIDER = _build_client()


def gemini_call_kwargs() -> dict:
    """Extra chat.completions.create kwargs needed only for the active provider."""
    return GEMINI_NO_THINKING_KWARGS if AI_PROVIDER == "gemini" else {}
