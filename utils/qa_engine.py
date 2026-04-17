"""
Intelligent Q&A Engine for Research Papers
Recognizes question intents and extracts specific information.
"""

import re
from typing import Dict, List, Tuple
from utils.vector_store import retrieve_relevant_chunks
from utils.summarizer import generate_answer

# Question intent templates
INTENT_PATTERNS = {
    'objective': {
        'patterns': [
            r'what.*objective|what.*goal|what.*aim|what.*propose|what.*paper.*about',
            r'objective.*of|goal.*of|main.*aim|purpose.*of.*study',
            r'what.*does.*this.*paper|what.*is.*main.*contribution'
        ],
        'search_keywords': 'objective goal aim contribution purpose propose main',
    },
    
    'algorithm': {
        'patterns': [
            r'what.*algorithm|what.*method|what.*approach|how.*do.*you|how.*does.*[a-z]+.*work',
            r'algorithm.*used|method.*proposed|approach.*presented|technique.*employed'
        ],
        'search_keywords': 'algorithm method approach technique model architecture proposed',
    },
    
    'dataset': {
        'patterns': [
            r'what.*dataset|which.*dataset|what.*data.*used|what.*benchmark',
            r'dataset.*used|data.*employed|benchmark.*dataset'
        ],
        'search_keywords': 'dataset data benchmark corpus evaluation test',
    },
    
    'results': {
        'patterns': [
            r'what.*result|what.*performance|what.*accuracy|how.*good|how.*well',
            r'result.*achieve|performance.*achieved|metric.*result|accuracy.*test',
            r'BLEU|F1|accuracy|precision|recall|AUC|ROUGE'
        ],
        'search_keywords': 'results performance metric accuracy F1 BLEU precision recall AUC',
    },
    
    'contribution': {
        'patterns': [
            r'what.*contribution|what.*novel|what.*new|contribution.*made',
            r'novel.*approach|new.*method|first.*to|main.*advantage'
        ],
        'search_keywords': 'contribution novel new contribution main advantage breakthrough',
    },
    
    'limitation': {
        'patterns': [
            r'what.*limitation|what.*drawback|what.*weakness|limit.*of',
            r'limitation.*discussed|weakness.*identified|constraint.*identified'
        ],
        'search_keywords': 'limitation drawback weakness constraint challenge future work',
    },
    
    'comparison': {
        'patterns': [
            r'how.*compare|compare.*with|vs|versus|better.*than|worse.*than',
            r'compared.*to|comparison.*with|baseline.*compare'
        ],
        'search_keywords': 'comparison compared baseline state-of-the-art SOTA versus',
    },
    
    'related_work': {
        'patterns': [
            r'related.*work|prior.*work|previous.*research|existing.*method',
            r'what.*exist|existing.*approach'
        ],
        'search_keywords': 'related work prior work previous research existing literature',
    },
    
    'experiment': {
        'patterns': [
            r'experimental.*setup|experiment.*design|how.*evaluate|evaluation.*method',
            r'experiment.*conducted|test.*setup|implementation.*detail'
        ],
        'search_keywords': 'experiment experimental setup evaluation method implementation detail',
    },
    
    'author': {
        'patterns': [
            r'who.*author|who.*wrote|author.*of|written.*by',
            r'affiliation|institution|university|organization'
        ],
        'search_keywords': 'author authors affiliation university institution organization',
    }
}

def detect_intent(question: str) -> Tuple[str, float]:
    """
    Detect the intent of a question.
    
    Returns:
        (intent_type, confidence_score)
    """
    question_lower = question.lower()
    best_intent = 'objective'
    best_score = 0
    
    for intent, config in INTENT_PATTERNS.items():
        for pattern in config['patterns']:
            match = re.search(pattern, question_lower)
            if match:
                score = 0.9 + (len(match.group(0)) / len(question_lower) * 0.1)
                if score > best_score:
                    best_score = score
                    best_intent = intent
    
    return best_intent, min(best_score, 1.0)


def answer_structured_question(text: str, question: str, chunks: List[str] = None, chat_history: List[Dict] = None, chunk_embeddings=None) -> Dict:
    """
    Answer a structured question about a research paper.
    
    Args:
        text: Full extracted text from PDF
        question: User's question
        chunks: Pre-chunked text (optional)
        chat_history: Optional list of previous dict messages for context resolution
    
    Returns:
        dict: Contains intent, answer, confidence, source snippets
    """
    from utils.vector_store import chunk_text
    from utils.summarizer import resolve_contextual_query
    
    # 1) If this is a follow-up, rewrite the query (e.g., "explain it deeper" -> "explain the GPT-3 objective deeper")
    if chat_history and len(chat_history) > 0:
        standalone_question = resolve_contextual_query(chat_history, question)
        print(f"[Context Resolution]\nOriginal: {question}\nResolved: {standalone_question}")
    else:
        standalone_question = question
        
    intent, _ = detect_intent(standalone_question)
    
    # Get chunks if not provided
    if chunks is None:
        chunks = chunk_text(text)
    
    # Use the specific context-aware question to find the best paragraphs
    search_query = standalone_question
    
    # Retrieve relevant chunks
    try:
        relevant_chunks, confidence = retrieve_relevant_chunks(chunks, search_query, return_confidence=True, chunk_embeddings=chunk_embeddings)
    except Exception as e:
        print(f"Retrieval Error: {e}")
        relevant_chunks = chunks[:5]
        confidence = 0.5
    
    # Extract paper abstract to ground the LLM
    paper_info = extract_paper_info(text)
    abstract = paper_info.get('abstract', '')
    
    # Generate answer
    base_context = " ".join(relevant_chunks) if relevant_chunks else text[:2000]
    
    if abstract and abstract not in base_context:
        context = f"Abstract: {abstract}\n\nExcerpts: {base_context}"
    else:
        context = base_context
        
    try:
        answer = generate_answer(context, question)
    except Exception:
        answer = "Unable to generate answer. Please try a different question."
    
    return {
        'question': question,
        'resolved_question': standalone_question if standalone_question != question else None,
        'intent': intent,
        'confidence': round(confidence, 3),
        'answer': answer,
        'source_count': len(relevant_chunks),
        'question_type': f"{intent.capitalize()} Question"
    }


def extract_paper_info(text: str) -> Dict:
    """
    Extract structured information from a research paper.
    
    Returns:
        dict: Contains title, abstract, authors, objectives, methods, etc.
    """
    info = {}
    
    # Extract abstract
    abstract_match = re.search(
        r'(?i)abstract\s*[\n:]\s*(.{50,500}?)(?=\n\n|introduction|1\.|keywords)',
        text,
        re.DOTALL
    )
    if abstract_match:
        info['abstract'] = abstract_match.group(1).strip()
    
    # Extract title (usually first substantial text or in metadata)
    lines = text.split('\n')
    for line in lines[:20]:
        if len(line.strip()) > 20 and len(line.strip()) < 200 and not re.search(r'^\s*\d+\s*$', line):
            if not any(x in line.lower() for x in ['page', 'abstract', 'introduction', 'copyright']):
                info['title'] = line.strip()
                break
    
    # Extract keywords
    keywords_match = re.search(
        r'(?i)keyword[s]?[\s:]*(.{10,200}?)(?=\n\n|abstract|introduction|©)',
        text,
        re.DOTALL
    )
    if keywords_match:
        keywords = keywords_match.group(1).strip().split(',')
        info['keywords'] = [k.strip() for k in keywords if k.strip()]
    
    # Extract main sections
    sections = {}
    section_pattern = r'(?i)^\s*(abstract|introduction|related work|method|algorithm|results|discussion|conclusion)\s*[\n:]'
    
    matches = list(re.finditer(section_pattern, text, re.MULTILINE))
    for i, match in enumerate(matches):
        section_name = match.group(1).lower()
        start = match.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        content = text[start:end].strip()[:500]  # First 500 chars of each section
        sections[section_name] = content
    
    if sections:
        info['sections'] = sections
    
    return info


def generate_qa_report(text: str, chunks: List[str] = None, chunk_embeddings=None) -> Dict:
    """
    Generate a comprehensive Q&A report about a paper.
    Answers common research paper questions automatically.
    
    Returns:
        dict: Contains answers to predefined questions
    """
    from utils.vector_store import chunk_text
    
    if chunks is None:
        chunks = chunk_text(text)
    
    # Predefined important questions
    important_questions = [
        "What is the main objective of this paper?",
        "What algorithm or method is proposed?",
        "What dataset was used for evaluation?",
        "What are the main results and performance metrics?",
        "What is the novel contribution of this work?",
        "What are the limitations of this approach?",
        "How does this compare with existing methods?",
        "What is the experimental setup?",
    ]
    
    report = {
        'paper_info': extract_paper_info(text),
        'qa_pairs': [],
        'summary': None
    }
    
    # Answer each question
    for question in important_questions:
        qa = answer_structured_question(text, question, chunks, chunk_embeddings=chunk_embeddings)
        report['qa_pairs'].append(qa)
    
    # Generate overall summary
    try:
        from utils.summarizer import generate_summary
        summary = generate_summary(text)
        report['summary'] = summary
    except Exception:
        pass
    
    return report


def format_qa_output(qa_result: Dict) -> str:
    """Format Q&A result for readable output."""
    output = []
    output.append(f"Q: {qa_result['question']}")
    output.append(f"   Type: {qa_result['question_type']} (Confidence: {qa_result['confidence']*100:.0f}%)")
    output.append(f"A: {qa_result['answer']}")
    output.append("")
    return "\n".join(output)


def format_report_output(report: Dict) -> str:
    """Format comprehensive report for readable output."""
    output = []
    
    # Paper info
    output.append("=" * 60)
    output.append("RESEARCH PAPER ANALYSIS REPORT")
    output.append("=" * 60)
    output.append("")
    
    if 'paper_info' in report and report['paper_info']:
        info = report['paper_info']
        if 'title' in info:
            output.append(f"[Title] {info['title']}")
        if 'abstract' in info:
            output.append(f"\n[Abstract]:\n{info['abstract'][:300]}...")
        if 'keywords' in info:
            output.append(f"\n[Keywords] {', '.join(info['keywords'][:5])}")
        output.append("\n" + "-" * 60 + "\n")
    
    # Q&A pairs
    output.append("KEY QUESTIONS & ANSWERS:")
    output.append("-" * 60)
    for i, qa in enumerate(report['qa_pairs'], 1):
        output.append(f"\n{i}. {qa['question_type'].upper()}")
        output.append(f"   Q: {qa['question']}")
        output.append(f"   A: {qa['answer'][:200]}..." if len(qa['answer']) > 200 else f"   A: {qa['answer']}")
    
    # Summary
    if report.get('summary'):
        output.append("\n" + "=" * 60)
        output.append("PAPER SUMMARY:")
        output.append(report['summary'][:500])
    
    return "\n".join(output)
