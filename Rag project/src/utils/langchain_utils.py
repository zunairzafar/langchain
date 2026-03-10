"""
LangChain-specific utilities — prompt templates, LLM factory, retriever builder.

Covers:
    - RAG QA prompt template      (Step 11)
    - LLM instantiation via HuggingFaceHub / HuggingFaceEndpoint
    - Vector-store → retriever    (Step 8-9)
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Load .env from the project root (Rag project/) regardless of CWD
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_RAG_QA_TEMPLATE = """\
You are a helpful assistant that answers questions based on the provided document context.
Use the chat history to understand follow-up questions.
If you cannot find the answer in the context, say "I don't have enough information to answer that."

Chat History:
{chat_history}

Context:
{context}

Question: {question}

Answer:"""

_QUERY_REWRITE_TEMPLATE = """\
Given the following chat history and a follow-up question, rewrite the follow-up question to be a standalone question that includes all necessary context.
If the question is already standalone, return it as-is. Only return the rewritten question, nothing else.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""


def get_qa_prompt_template() -> PromptTemplate:
    """Return the default RAG question-answering prompt template."""
    return PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template=_RAG_QA_TEMPLATE,
    )


def get_query_rewrite_template() -> PromptTemplate:
    """Return template that rewrites follow-up questions into standalone queries."""
    return PromptTemplate(
        input_variables=["chat_history", "question"],
        template=_QUERY_REWRITE_TEMPLATE,
    )


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def get_llm(
    model_name: str | None = None,
    temperature: float = 0.1,
    max_new_tokens: int = 512,
    **kwargs,
):
    """Instantiate and return a ChatHuggingFace LLM via HuggingFaceEndpoint.

    Requires HF_TOKEN in environment / .env file.
    """
    model_name = model_name or os.getenv(
        "LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct"
    )
    llm = HuggingFaceEndpoint(
        repo_id=model_name,
        task="text-generation",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        **kwargs,
    )
    return ChatHuggingFace(llm=llm)


# ---------------------------------------------------------------------------
# Retriever helper
# ---------------------------------------------------------------------------

def get_retriever(vector_store, search_type: str = "similarity", search_kwargs: dict | None = None):
    """Wrap a FAISS / Chroma vector store in a LangChain retriever."""
    search_kwargs = search_kwargs or {"k": 4}
    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )
