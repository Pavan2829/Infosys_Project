"""
Enhanced Streamlit Web Interface for Research Paper Summarizer
With intelligent Q&A for specific research paper questions
"""

import os
import json
import tempfile
import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(__file__))

from utils.pdf_extractor import extract_text_from_pdf
from utils.vector_store import chunk_text
from utils.rag_pipeline import rag_summarize, rag_answer
from utils.evaluator import evaluate_llm
from utils.qa_engine import (
    answer_structured_question,
    generate_qa_report,
    format_report_output,
    detect_intent,
    INTENT_PATTERNS
)

st.set_page_config(
    page_title="AI-Powered Research Paper Summarizer & Insight Extractor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("AI-Powered Research Paper Summarizer & Insight Extractor")
st.markdown("Upload research papers and get instant summaries, answers to specific questions, and detailed analysis.")

with st.sidebar:
    st.header("Settings")
    
    mode = st.radio(
        "Select Mode",
        ["Summarize", "Ask Questions", "Full Analysis", "Q&A Report"],
        help="Choose how you want to interact with the paper"
    )
    
    st.markdown("---")
    if st.button("⚠️ Clear Cache & Restart", help="Click this if answers seem stuck or outdated from a previous run!"):
        st.session_state.clear()
        st.rerun()

    st.markdown("### Question Types Supported")
    st.markdown("""
    - **Objective**: Main goal and contribution
    - **Algorithm**: Methods and techniques
    - **Dataset**: Data and benchmarks used
    - **Results**: Performance and metrics
    - **Contribution**: Novel aspects
    - **Limitation**: Weaknesses/constraints
    - **Comparison**: vs existing methods
    - **Experiment**: Setup and process
    """)

uploaded_file = st.file_uploader(
    "Choose a research paper PDF",
    type=["pdf"],
    help="Upload any research paper (max 200MB)"
)

if uploaded_file is not None:
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_pdf_path = tmp.name
    
    st.success(f"[OK] Loaded: {uploaded_file.name}")
    # Cache Buster: Compare current file with lastly processed file to forcibly flush RAM.
    if 'last_uploaded' not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
        st.session_state.clear()
        st.session_state.last_uploaded = uploaded_file.name
        st.session_state.messages = []
        st.rerun()

    # Re-initialize history on manual cache clears
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Extract once and cache
    if 'text' not in st.session_state:
        with st.spinner("[Processing] PDF..."):
            st.session_state.text = extract_text_from_pdf(temp_pdf_path)
            st.session_state.chunks = chunk_text(st.session_state.text)
    
    text = st.session_state.text
    chunks = st.session_state.chunks
    
    st.info(f"[Metrics] Extracted: {len(text):,} characters | {len(chunks)} chunks")
    
    # MODE 1: SUMMARIZE
    if mode == "Summarize":
        st.header("Generate Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("[Run] Generate Summary", key="gen_summary"):
                with st.spinner("Generating optimal summary..."):
                    summary = rag_summarize(temp_pdf_path)
                    st.session_state.summary = summary
        
        if 'summary' in st.session_state:
            st.subheader("Generated Summary")
            st.write(st.session_state.summary)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Text Length", f"{len(text):,}")
            with col2:
                st.metric("Summary Length", f"{len(st.session_state.summary):,}")
            with col3:
                ratio = len(st.session_state.summary) / len(text)
                st.metric("Compression", f"{ratio:.1%}")
            
            # Evaluation option
            with col2:
                if st.checkbox("[Metrics] Include Evaluation"):
                    reference = st.text_area(
                        "Paste reference summary",
                        height=150,
                        key="ref_for_summary"
                    )
                    
                    if reference and st.button("[Evaluate]"):
                        with st.spinner("Evaluating..."):
                            scores = evaluate_llm(st.session_state.summary, reference)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("ROUGE-1", f"{scores['rouge1']['f1']:.3f}")
                            with col2:
                                st.metric("ROUGE-2", f"{scores['rouge2']['f1']:.3f}")
                            with col3:
                                if 'bertscore' in scores:
                                    st.metric("BERTScore", f"{scores['bertscore']['f1']:.3f}")
            
            st.download_button(
                "[Download] Summary",
                st.session_state.summary,
                f"summary_{uploaded_file.name.replace('.pdf', '.txt')}",
                "text/plain"
            )
    
    # MODE 2: ASK QUESTIONS (Chat Interface)
    elif mode == "Ask Questions":
        st.header("💬 Chat with the Paper")
        st.markdown("Ask anything about the uploaded document. The AI will find the answer and cite its sources!")
        
        # Initialize with a greeting if empty
        if len(st.session_state.messages) == 0:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Hello! 👋 I've just read **{uploaded_file.name}**. What would you like to know about it?"
            })
            
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            avatar = "🤖" if message["role"] == "assistant" else "👤"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("Message the AI Assistant..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant", avatar="🤖"):
                message_placeholder = st.empty()
                with st.spinner("Thinking..."):
                    # Generate Answer passing history
                    history_for_context = st.session_state.messages[:-1] # exclude the current prompt just appended
                    qa_result = answer_structured_question(text, prompt, chunks, chat_history=history_for_context)
                    
                    if qa_result.get('answer'):
                        answer_text = qa_result['answer']
                        
                        # Add extra metadata as a formatted string
                        conf = qa_result.get('confidence', 0)
                        
                        resolved_msg = f"*(Resolved Context: '{qa_result['resolved_question']}')*\n\n" if qa_result.get('resolved_question') else ""
                        sources = f"\n\n---\n*Sources: {qa_result.get('source_count', 0)} chunks | Confidence: {conf*100:.0f}%*"
                        full_response = resolved_msg + answer_text + sources
                        
                        message_placeholder.markdown(full_response)
                        # Add to session state
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                    else:
                        error_msg = "I couldn't find an answer to that. Could you rephrase your question?"
                        message_placeholder.markdown(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # MODE 3: FULL ANALYSIS
    elif mode == "Full Analysis":
        st.header("Comprehensive Paper Analysis")
        
        if st.button("[Generate] Full Analysis", key="full_analysis"):
            with st.spinner("Analyzing paper..."):
                report = generate_qa_report(text, chunks)
                st.session_state.report = report
        
        if 'report' in st.session_state:
            report = st.session_state.report
            
            # Paper Info
            if 'paper_info' in report and report['paper_info']:
                with st.expander("[Info] Paper Information", expanded=True):
                    info = report['paper_info']
                    if 'title' in info:
                        st.markdown(f"**Title**: {info['title']}")
                    if 'abstract' in info:
                        st.markdown(f"**Abstract**: {info['abstract']}")
                    if 'keywords' in info:
                        st.markdown(f"**Keywords**: {', '.join(info['keywords'][:10])}")
            
            # Q&A Pairs
            st.markdown("### Questions & Answers")
            for i, qa in enumerate(report['qa_pairs'], 1):
                with st.expander(f"{i}. {qa['question_type'].upper()}", expanded=(i<=3)):
                    st.markdown(f"**Q**: {qa['question']}")
                    st.markdown(f"**A**: {qa['answer']}")
                    st.caption(f"Confidence: {qa['confidence']*100:.0f}% | Sources: {qa['source_count']}")
            
            # Summary
            if report.get('summary'):
                with st.expander("[Summary] Paper Summary"):
                    st.write(report['summary'])
    
    # MODE 4: Q&A REPORT
    elif mode == "Q&A Report":
        st.header("Structured Q&A Report")
        
        if st.button("[Report] Generate Q&A Report", key="qa_report"):
            with st.spinner("Generating report..."):
                report = generate_qa_report(text, chunks)
                
                # Display as formatted text
                report_text = format_report_output(report)
                st.text(report_text)
                
                # Download options
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "[Download] Text",
                        report_text,
                        f"report_{uploaded_file.name.replace('.pdf', '.txt')}",
                        "text/plain"
                    )
                
                with col2:
                    st.download_button(
                        "[Download] JSON",
                        json.dumps(report, indent=2),
                        f"report_{uploaded_file.name.replace('.pdf', '.json')}",
                        "application/json"
                    )
    
    # Clean up
    try:
        os.unlink(temp_pdf_path)
    except:
        pass

else:
    st.info("Upload a research paper PDF to get started!")
    
    st.markdown("---")
    st.markdown("""
    ## Features
    
    ### Summarize
    - Generate abstractive summaries using local Ollama
    - Evaluate against reference summaries
    - Compare with original using ROUGE & BERTScore
    
    ### Ask Questions
    - Answer specific research questions
    - Detect question intent automatically
    - Get cited sources for answers
    
    ### Full Analysis
    - Extract paper metadata
    - Answer 8+ key research questions
    - Generate comprehensive reports
    
    ### Q&A Reports
    - Structured question-answer pairs
    - Export as text or JSON
    - Share research insights
    
    ---
    
    ## Models Used
    
    | Component | Model | Purpose |
    |-----------|-------|---------|
    | Summarization | Local Ollama | Secure abstractive summaries |
    | Embeddings | All-MPNet-Base-v2 | Semantic understanding |
    | Re-ranking | Cross-Encoder MS-Marco | Ranking relevance |
    | Evaluation | ROUGE + BERTScore | Quality assessment |
    
    ---
    
    Made with care for researchers | v3.0 with SmartQA
    """)

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>[Secure] Your documents are processed locally and never stored</p>", unsafe_allow_html=True)
