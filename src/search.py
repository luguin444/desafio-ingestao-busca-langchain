import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_postgres import PGVector

from providers_helper import get_embedding_model, get_llm

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")
TOP_K = 10 # number of documents to retrieve from the vector store

PROMPT_TEMPLATE = """
CONTEXTO:
{context}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{question}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def _get_store() -> PGVector:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not defined")
    if not COLLECTION_NAME:
        raise RuntimeError("PG_VECTOR_COLLECTION_NAME not defined")

    return PGVector(
        embeddings=get_embedding_model(),
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )


def search_prompt(question: str) -> str:
    store = _get_store()
    results = store.similarity_search_with_score(question, k=TOP_K)
    context = "\n\n".join(doc.page_content for doc, _score in results)

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | get_llm()
    response = chain.invoke({"context": context, "question": question})
    return response.content
