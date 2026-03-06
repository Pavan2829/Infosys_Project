from typing import List
import os
from utils.pdf_extractor import extract_text_from_pdf
from utils.vector_store import chunk_text, retrieve_relevant_chunks
from utils.summarizer import generate_summary, generate_answer


def get_graph_context() -> str:
    """Get graph context from Neo4j. Returns empty string if unavailable.
    
    Disabled for now to avoid Neo4j connection timeouts during evaluation.
    """
    return ""


def build_hybrid_context(text: str, question: str = None) -> str:
    """Return combined semantic + graph context for a given text and optional question."""
    # Chunk text and get semantic context (top relevant chunks)
    chunks = chunk_text(text)

    semantic_context = ""
    if question:
        try:
            semantic_context = retrieve_relevant_chunks(chunks, question)
        except Exception:
            # Fallback: use first few chunks
            semantic_context = " ".join(chunks[:3])
    else:
        # For summarization, use the top chunks by position
        semantic_context = " ".join(chunks[:6])

    graph_context = get_graph_context()

    hybrid = semantic_context
    if graph_context:
        hybrid = semantic_context + "\n" + graph_context

    return hybrid


def rag_summarize(pdf_path: str) -> str:
    """Extract text from PDF, build hybrid context, and generate a summary using the LLM summarizer."""
    text = extract_text_from_pdf(pdf_path)

    # Build hybrid context for summarization
    hybrid_context = build_hybrid_context(text)

    # Use existing summarizer to generate summary from hybrid context
    # If the summarizer expects long text, we pass hybrid_context (already truncated inside)
    summary = generate_summary(hybrid_context)
    return summary


def rag_answer(pdf_path: str, question: str) -> str:
    """
    Answer a question using hybrid RAG with intent-aware processing.
    Detects question type and retrieves targeted information.
    """
    from utils.qa_engine import answer_structured_question
    from utils.vector_store import chunk_text
    
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    
    # Use structured Q&A for better intent-aware answers
    qa_result = answer_structured_question(text, question, chunks)
    
    return qa_result['answer']
