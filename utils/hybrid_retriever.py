from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
from neo4j import GraphDatabase

# Upgraded to better embedding model
model = SentenceTransformer('all-mpnet-base-v2')

# Cross-encoder for re-ranking
try:
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
except Exception:
    reranker = None

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j","nani1234")
)

def retrieve_graph():

    query = """
    MATCH (a)-[r]->(b)
    RETURN a.name AS subject,type(r) AS relation,b.name AS object
    LIMIT 20
    """

    with driver.session() as session:
        result = session.run(query)

        graph = ""

        for record in result:
            graph += f"{record['subject']} {record['relation']} {record['object']}. "

    return graph


def hybrid_retrieve(question, index, chunks):
    """
    Retrieve using semantic embeddings and re-rank with cross-encoder.
    Combines embeddings with knowledge graph for better context.
    """
    q_embed = model.encode([question])

    # Get more candidates (10 instead of 5)
    D, I = index.search(np.array(q_embed), 10)

    semantic_chunks = [chunks[i] for i in I[0]]
    
    # Re-rank with cross-encoder if available
    if reranker is not None:
        try:
            pairs = [[question, chunk] for chunk in semantic_chunks]
            scores = reranker.predict(pairs)
            reranked_indices = np.argsort(scores)[::-1][:5]
            semantic_chunks = [semantic_chunks[i] for i in reranked_indices]
        except Exception:
            semantic_chunks = semantic_chunks[:5]
    else:
        semantic_chunks = semantic_chunks[:5]
    
    semantic_context = " ".join(semantic_chunks)

    graph_context = retrieve_graph()

    hybrid_context = semantic_context
    if graph_context:
        hybrid_context = semantic_context + "\n" + graph_context

    return hybrid_context, q_embed