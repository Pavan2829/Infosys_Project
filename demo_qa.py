"""
Demo Script: Research Paper SmartQA
Shows how the Q&A engine answers different types of research questions
"""

import os
import json
from utils.pdf_extractor import extract_text_from_pdf
from utils.vector_store import chunk_text
from utils.qa_engine import (
    answer_structured_question,
    detect_intent,
    INTENT_PATTERNS,
    extract_paper_info
)

def demo_paper_analysis(pdf_path: str):
    """
    Demonstrate the SmartQA system on a research paper.
    """
    
    if not os.path.exists(pdf_path):
        print(f"[ERROR] PDF not found: {pdf_path}")
        return
    
    pdf_name = os.path.basename(pdf_path)
    
    print("\n" + "="*70)
    print(f"RESEARCH PAPER SMARTQA DEMO: {pdf_name}")
    print("="*70)
    
    # Extract
    print("\nExtracting and processing paper...")
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    print(f"   [OK] Extracted {len(text):,} characters")
    print(f"   [OK] Created {len(chunks)} semantic chunks")
    
    # Extract metadata
    paper_info = extract_paper_info(text)
    
    if 'title' in paper_info:
        print(f"\n[Title] {paper_info['title'][:80]}")
    
    if 'abstract' in paper_info:
        abstract_preview = paper_info['abstract'][:200].replace('\n', ' ')
        print(f"[Abstract] {abstract_preview}...")
    
    if 'keywords' in paper_info:
        print(f"[Keywords] {', '.join(paper_info['keywords'][:5])}")
    
    # Demo questions
    demo_questions = [
        ("What is the main objective of this paper?", "objective"),
        ("What algorithm or method is proposed?", "algorithm"),
        ("What dataset was used for evaluation?", "dataset"),
        ("What are the main results?", "results"),
        ("What is the novel contribution?", "contribution"),
    ]
    
    print("\n" + "="*70)
    print("DEMONSTRATING SMARTQA - ANSWERING RESEARCH QUESTIONS")
    print("="*70)
    
    for i, (question, expected_intent) in enumerate(demo_questions, 1):
        print(f"\n[{i}] Question: {question}")
        
        # Detect intent
        detected_intent, confidence = detect_intent(question)
        print(f"    [Intent] {detected_intent.upper()} | Confidence: {confidence*100:.0f}%")
        print(f"    [Match] Expected: {expected_intent.upper()} | Result: {'OK' if detected_intent == expected_intent else 'WARNING'}")
        
        # Get answer
        print(f"    [Retrieving answer...]")
        qa_result = answer_structured_question(text, question, chunks)
        
        # Display answer
        answer_preview = qa_result['answer'][:150].replace('\n', ' ')
        if len(qa_result['answer']) > 150:
            answer_preview += "..."
        
        # safely encode/decode for Windows console display
        safe_answer = answer_preview.encode('ascii', 'ignore').decode('ascii')
        print(f"\n    [Answer] {safe_answer}")
        print(f"    [Sources] {qa_result['source_count']} chunks | Confidence: {qa_result['confidence']*100:.0f}%")
    
    # Show available question types
    print("\n" + "="*70)
    print("AVAILABLE QUESTION TYPES IN SMARTQA")
    print("="*70)
    
    for intent, config in INTENT_PATTERNS.items():
        example_pattern = config['patterns'][0]
        print(f"\n- {intent.upper()}")
        print(f"  Pattern: {example_pattern}")
        print(f"  Keywords: {config['search_keywords'][:50]}...")
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("""
Try these in your own session:

  # Interactive mode (recommended for exploring)
  python ask_questions.py --pdf "paper.pdf" --interactive
  
  # Quick single question
  python ask_questions.py --pdf "paper.pdf" --question "What is the contribution?"
  
  # Generate full report
  python ask_questions.py --pdf "paper.pdf" --report
  
  # Streamlit web interface
  streamlit run app_new.py
    """)

if __name__ == '__main__':
    import sys
    
    # Use first PDF in data/input if available, otherwise ask for path
    default_pdf = "data/input/attention is all you need.pdf"
    
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else default_pdf
    
    demo_paper_analysis(pdf_path)
