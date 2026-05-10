import os

from dotenv import load_dotenv

load_dotenv()

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", EMBEDDING_PROVIDER).lower()


def get_embedding_model():
    if EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        )

    if EMBEDDING_PROVIDER in ("google", "gemini"):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        # Truncate to 768 dims via MRL — much smaller storage/faster search with minimal quality loss.
        return GoogleGenerativeAIEmbeddings(
            model=os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001"),
            output_dimensionality=768,
        )

    raise ValueError(
        f"invalid EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER!r}. Use 'openai' or 'google'."
    )


def get_llm():
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_LLM_MODEL", "gpt-5-nano"),
            temperature=0,
        )

    if LLM_PROVIDER in ("google", "gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash-lite"),
            temperature=0,
        )

    raise ValueError(
        f"invalid LLM_PROVIDER: {LLM_PROVIDER!r}. Use 'openai' or 'google'."
    )
