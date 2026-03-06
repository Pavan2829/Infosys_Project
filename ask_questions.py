"""
Interactive Research Paper Assistant
Allows users to ask specific questions about research papers.
Features: Objective, Algorithm, Dataset, Results, etc.
"""

import os
import json
import argparse
from pathlib import Path
from utils.pdf_extractor import extract_text_from_pdf
from utils.vector_store import chunk_text
from utils.qa_engine import (
    answer_structured_question, 
    generate_qa_report,
    format_qa_output,
    format_report_output,
    detect_intent,
    extract_paper_info,
    INTENT_PATTERNS
)


def interactive_qa(pdf_path: str):
    """
    Interactive Q&A session with a research paper.
    User can ask multiple questions about the paper.
    """
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return
    
    pdf_name = os.path.basename(pdf_path)
    print(f"\n{'='*60}")
    print(f"Research Paper Assistant: {pdf_name}")
    print(f"{'='*60}")
    
    # Extract and process
    print("\nProcessing paper...")
    try:
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)
        print(f"[OK] Extracted {len(text):,} characters")
        print(f"[OK] Created {len(chunks)} semantic chunks")
    except Exception as e:
        print(f"[ERROR] Error processing PDF: {str(e)}")
        return
    
    # Extract paper info
    paper_info = extract_paper_info(text)
    if 'title' in paper_info:
        print(f"\n[Title] {paper_info['title'][:100]}")
    if 'abstract' in paper_info:
        print(f"[Abstract] {paper_info['abstract'][:150]}...")
    
    print(f"\n{'='*60}")
    print("AVAILABLE QUESTION TYPES:")
    print(f"{'='*60}")
    for intent, config in INTENT_PATTERNS.items():
        print(f"  - {intent.upper()}: {', '.join(config['patterns'][:1])}")
    
    print(f"\n{'='*60}")
    print("Ask questions (type 'quit' to exit, 'report' for full report)")
    print(f"{'='*60}\n")
    
    while True:
        try:
            question = input("❓ Your question: ").strip()
            
            if not question:
                continue
            
            if question.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            if question.lower() == 'report':
                print("\n🔄 Generating comprehensive report...")
                report = generate_qa_report(text, chunks)
                print(format_report_output(report))
                continue
            
            # Detect intent
            intent, confidence = detect_intent(question)
            print(f"   🎯 Detected: {intent.upper()} (confidence: {confidence*100:.0f}%)\n")
            
            # Answer question
            print("   🔄 Finding answer...")
            qa_result = answer_structured_question(text, question, chunks)
            
            # Format and display
            print(format_qa_output(qa_result))
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"[ERROR] {str(e)}\n")


def batch_qa(pdf_dir: str, output_file: str = None):
    """
    Process multiple papers and generate Q&A reports for all.
    
    Args:
        pdf_dir: Directory containing PDFs
        output_file: JSON file to save results
    """
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"[ERROR] No PDFs found in {pdf_dir}")
        return
    
    print(f"\nFound {len(pdf_files)} PDF(s)")
    
    all_reports = {}
    
    for i, pdf_name in enumerate(pdf_files, 1):
        pdf_path = os.path.join(pdf_dir, pdf_name)
        print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_name}")
        
        try:
            text = extract_text_from_pdf(pdf_path)
            chunks = chunk_text(text)
            
            print(f"    [OK] Generating Q&A report...")
            report = generate_qa_report(text, chunks)
            all_reports[pdf_name] = report
            
            # Display quick summary
            if report['qa_pairs']:
                print(f"    [OK] Generated {len(report['qa_pairs'])} Q&A pairs")
            
        except Exception as e:
            print(f"    [ERROR] {str(e)}")
            all_reports[pdf_name] = {'error': str(e)}
    
    # Save results
    if output_file is None:
        output_file = os.path.join(pdf_dir, "qa_reports.json")
    
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_reports, f, indent=2)
    
    print(f"\n[SUCCESS] Reports saved to: {output_file}")
    print(f"   Processed: {len([r for r in all_reports.values() if 'qa_pairs' in r])}/{len(pdf_files)} PDFs")


def quick_qa(pdf_path: str, questions: list = None):
    """
    Quick Q&A without interactive prompts.
    
    Args:
        pdf_path: Path to PDF
        questions: List of questions to answer
    """
    if not os.path.exists(pdf_path):
        print(f"[ERROR] File not found: {pdf_path}")
        return
    
    if questions is None:
        # Default important questions
        questions = [
            "What is the main objective of this paper?",
            "What algorithm or method is proposed?",
            "What are the main results?",
        ]
    
    pdf_name = os.path.basename(pdf_path)
    print(f"\n{pdf_name}")
    print("=" * 60)
    
    try:
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)
        
        for question in questions:
            qa_result = answer_structured_question(text, question, chunks)
            print(format_qa_output(qa_result))
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Research Paper Q&A Assistant')
    parser.add_argument('--pdf', type=str, help='Path to single PDF')
    parser.add_argument('--dir', type=str, help='Directory with PDFs')
    parser.add_argument('--question', type=str, help='Specific question to answer')
    parser.add_argument('--interactive', action='store_true', help='Start interactive session')
    parser.add_argument('--report', action='store_true', help='Generate full report')
    parser.add_argument('--output', type=str, help='Output file for batch results')
    
    args = parser.parse_args()
    
    if args.pdf:
        if args.interactive or not args.question:
            # Interactive mode
            interactive_qa(args.pdf)
        elif args.question:
            # Quick Q&A mode
            quick_qa(args.pdf, [args.question])
        elif args.report:
            # Generate report
            text = extract_text_from_pdf(args.pdf)
            chunks = chunk_text(text)
            report = generate_qa_report(text, chunks)
            print(format_report_output(report))
    
    elif args.dir:
        # Batch processing
        batch_qa(args.dir, args.output)
    
    else:
        # Interactive with first PDF in data/input
        input_dir = os.path.join(os.getcwd(), "data", "input")
        pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')] if os.path.exists(input_dir) else []
        
        if pdf_files:
            pdf_path = os.path.join(input_dir, pdf_files[0])
            interactive_qa(pdf_path)
        else:
            print("Usage:")
            print("  python ask_questions.py --pdf <path/to/file.pdf> --interactive")
            print("  python ask_questions.py --pdf <path/to/file.pdf> --question 'your question'")
            print("  python ask_questions.py --pdf <path/to/file.pdf> --report")
            print("  python ask_questions.py --dir <directory> --output results.json")
