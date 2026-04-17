from sentence_transformers import SentenceTransformer, CrossEncoder  # type: ignore
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# -------------------------------
# LOAD EMBEDDING MODEL
# -------------------------------
# Downgraded to a much faster MiniLM model to improve the Q&A latency.
model = SentenceTransformer("all-MiniLM-L6-v2")

# Cross-encoder for re-ranking retrieved chunks
try:
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
except Exception:
    reranker = None


# -------------------------------
# STEP 1 : FILTER IMPORTANT PARTS
# -------------------------------
def extract_research_content(text):

    lower_text = text.lower()

    if "abstract" in lower_text:
        start = lower_text.find("abstract")
        text = text[start:]

    elif "introduction" in lower_text:
        start = lower_text.find("introduction")
        text = text[start:]
        
    # Strip references to prevent retrieving random citation titles
    ref_idx = -1
    for keyword in ['\nreferences', '\nbibliography']:
        idx = text.lower().rfind(keyword)
        if idx != -1:
             ref_idx = max(ref_idx, idx)
    
    if ref_idx != -1:
         text = text[:ref_idx]

    return text


# -------------------------------
# STEP 2 : SEMANTIC CHUNKING
# -------------------------------
def chunk_text(text, target_chunk_size=500, sentence_overlap=1):
    """
    Split text into semantic chunks using a sliding window of sentences.
    This guarantees chunks will never break grammar or words in half.
    """
    clean_text = extract_research_content(text)
    
    # Split into discrete sentences
    try:
        sentences = sent_tokenize(clean_text)
    except Exception:
        sentences = clean_text.split('. ')
    
    if not sentences:
        return [clean_text]
        
    chunks = []
    current_chunk_sentences = []
    current_length = 0
    
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_len = len(sentence)
        
        # If adding this sentence exceeds ideal size, save the current chunk
        if current_length + sentence_len > target_chunk_size and current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
            
            # Start new chunk with overlapping sentences (sliding window)
            overlap_start = max(0, len(current_chunk_sentences) - sentence_overlap)
            current_chunk_sentences = current_chunk_sentences[overlap_start:]
            current_length = sum(len(s) for s in current_chunk_sentences)
            
        current_chunk_sentences.append(sentence)
        current_length += sentence_len
        
    # Append the final remaining chunk
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))
        
    return chunks


# -------------------------------
# STEP 3 : VECTOR RETRIEVAL WITH RE-RANKING
# -------------------------------
def get_chunk_embeddings(chunks):
    """
    Pre-compute embeddings for all chunks so they don't have to be recalculated
    on every single user question.
    """
    if not chunks:
        return np.array([])
    return model.encode(chunks)

def retrieve_relevant_chunks(chunks, question, return_confidence=False, chunk_embeddings=None):
    """
    Retrieve relevant chunks using embeddings and re-rank with cross-encoder.
    This two-stage retrieval improves accuracy significantly.
    """
    if not chunks:
        if return_confidence:
            return [], 0.0
        return []

    # Stage 1: Semantic retrieval - get more candidates
    if chunk_embeddings is None:
        chunk_embeddings = model.encode(chunks)
    question_embedding = model.encode([question])

    similarities = np.dot(chunk_embeddings, question_embedding.T).flatten()

    # Retrieve more candidates for re-ranking (top 10 instead of 3)
    k = min(10, len(similarities))
    top_k_indices = similarities.argsort()[-k:][::-1]
    
    top_confidence = float(similarities[top_k_indices[0]]) if k > 0 else 0.0

    # Stage 2: Re-rank with cross-encoder if available
    if reranker is not None:
        candidate_chunks = [chunks[i] for i in top_k_indices]
        try:
            # Create pairs of (question, chunk) for cross-encoder
            pairs = [[question, chunk] for chunk in candidate_chunks]
            scores = reranker.predict(pairs)
            # Re-sort by cross-encoder scores
            reranked_indices = np.argsort(scores)[::-1]
            top_k_indices = [top_k_indices[i] for i in reranked_indices[:3]]
            
            # Use the cross-encoder purely for ranking. 
            # For the UI confidence score, use a boosted sigmoid curve on the base model's cosine similarity.
            # Raw cosine similarities for good matches often hover around 0.4-0.6. This math pushes those
            # valid matches up into the 80%+ range which feels much more intuitive to end users.
            top_best_idx = top_k_indices[0]
            raw_cosine = float(similarities[top_best_idx])
            
            # Non-linear boost: a raw score of 0.35 becomes ~80%, 0.55 becomes ~95%
            boosted_score = 1.0 / (1.0 + np.exp(-10 * (raw_cosine - 0.25)))
            top_confidence = float(np.clip(boosted_score, 0.0, 1.0))
        except Exception:
            # Fallback to original ranking if re-ranking fails
            top_k_indices = top_k_indices[:3]
    else:
        top_k_indices = top_k_indices[:3]
        
        # Apply the same non-linear boost when cross-encoder is disabled
        raw_cosine = float(similarities[top_k_indices[0]]) if k > 0 else 0.0
        boosted_score = 1.0 / (1.0 + np.exp(-10 * (raw_cosine - 0.25)))
        top_confidence = float(np.clip(boosted_score, 0.0, 1.0)) if k > 0 else 0.0

    relevant_chunks = [chunks[i] for i in top_k_indices]

    if return_confidence:
        return relevant_chunks, top_confidence
    return relevant_chunks