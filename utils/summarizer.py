import os
import json
import requests

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"

# -------------------------------
# GENERATE SUMMARY
# -------------------------------
def generate_summary(text):
    """
    Generate an abstractive summary using local Ollama instance.
    """
    try:
        prompt = f"Please write a concise summary of the following research paper text. Focus on the main goals and results.\n\nText:\n{text[:4000]}"
        
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3
            }
        }
        
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        return result.get('response', 'Error: No response generated from Ollama.').strip()
        
    except Exception as e:
        return f"Error generating summary via Ollama: {str(e)}"

# -------------------------------
# GENERATE ANSWER
# -------------------------------
def generate_answer(context, question):
    """
    Generate an answer based on context and question using a generative LLM via Ollama.
    """
    try:
        # Build strict prompt for generative LLM avoiding hallucinations
        prompt = f"Please answer the following question strictly using ONLY the provided context.\n\nContext block:\n{context[:4000]}\n\nQuestion asked: {question}\n\nDirect Answer:"
        
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1
            }
        }
        
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        return result.get('response', 'Error: No response generated from Ollama.').strip()
        
    except Exception as e:
        return f"Unable to generate answer via Ollama: {str(e)}"

# -------------------------------
# CONTEXTUAL QUERY RESOLUTION
# -------------------------------
def resolve_contextual_query(chat_history: list, current_question: str) -> str:
    """
    Rewrite a conversational query (like 'explain it more') into a standalone 
    semantic question using the previous chat history context.
    """
    if not chat_history:
        return current_question
        
    try:
        # Format the recent history for the LLM
        history_text = ""
        for msg in chat_history[-4:]: # Only need the last few back-and-forths
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content'][:500]}\n"
            
        prompt = f"""
Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question that can be understood without the conversation history. 
If the follow-up question is already standalone, just repeat it exactly as is.
DO NOT answer the question. ONLY return the rewritten standalone question.

Chat History:
{history_text}

Follow-up Question: {current_question}

Standalone Question:"""

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0 # Strict determinism for rewriting
            }
        }
        
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        standalone_q = result.get('response', current_question).strip()
        
        # Fallback in case the LLM tries to answer the question anyway
        if len(standalone_q) > len(current_question) * 4 and len(standalone_q) > 200:
            return current_question
            
        return standalone_q
        
    except Exception:
        return current_question
