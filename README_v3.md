# 📄 Research Paper Summarizer - Complete Guide

## 🎯 Project Overview

An AI-powered research paper analysis system that can:
- ✅ Generate high-quality summaries of research papers
- ✅ Answer specific questions about papers (objective, algorithm, dataset, results, etc.)
- ✅ Extract structured information automatically
- ✅ Evaluate summaries with ROUGE and BERTScore metrics
- ✅ Create comprehensive Q&A reports
- ✅ Work with **ANY research paper** users upload

---

## 🚀 Quick Start

### 1. **Interactive Web Interface**
```bash
streamlit run app_new.py
```
Browser opens at `http://localhost:8501`

### 2. **Command Line - Ask Questions**
```bash
# Interactive Q&A session
python ask_questions.py --pdf "path/to/paper.pdf" --interactive

# Ask specific question
python ask_questions.py --pdf "path/to/paper.pdf" --question "What is the main objective?"

# Generate full report
python ask_questions.py --pdf "path/to/paper.pdf" --report

# Batch process directory
python ask_questions.py --dir "data/input" --output "results.json"
```

### 3. **Complete Pipeline**
```bash
# Process all PDFs in data/input to generate metadata, summaries, and knowledge graphs
python run_project.py
```

---

## 🤖 Available Question Types

The system intelligently detects and answers:

### **1. Objective Question**
- "What is the main objective of this paper?"
- "What is the goal of this work?"
- "What problem does this solve?"

✨ **Answer**: Extracts main goal, contribution, and purpose

### **2. Algorithm Question**
- "What algorithm is used?"
- "What method is proposed?"
- "How does the approach work?"

✨ **Answer**: Describes techniques, models, and architectures

### **3. Dataset Question**
- "What dataset was used?"
- "Which benchmark was evaluated?"
- "What data is used for training?"

✨ **Answer**: Lists datasets, benchmarks, and data specifications

### **4. Results Question**
- "What are the main results?"
- "What metrics were achieved?"
- "How good is the performance?"

✨ **Answer**: Extracts performance metrics and comparisons

### **5. Contribution Question**
- "What is novel about this work?"
- "What is the main contribution?"
- "What is new?"

✨ **Answer**: Highlights novel aspects and breakthroughs

### **6. Limitation Question**
- "What are the limitations?"
- "What are the weaknesses?"
- "What future work is needed?"

✨ **Answer**: Identifies constraints and future directions

### **7. Comparison Question**
- "How does this compare with existing methods?"
- "Is it better than previous work?"
- "How does it compare with SOTA?"

✨ **Answer**: Compares with baselines and state-of-the-art

### **8. Experiment Question**
- "How was the experiment set up?"
- "What is the evaluation methodology?"
- "How was this tested?"

✨ **Answer**: Explains experimental design and setup

---

## 📊 Models & Components

### **Summarization & GenAI**
- **Model**: Local Ollama (llama3.2:1b default)
- **Features**: 
  - Absolutely local and secure abstractive summarization
  - Robust question-answering based purely on document context
  - Easily swappable via `OLLAMA_MODEL` environment variable

### **Embeddings & Retrieval**
- **Embedding Model**: All-MPNet-Base-v2 (better than MiniLM)
- **Re-ranker**: Cross-Encoder MS-Marco
- **Process**:
  1. Semantic retrieval (get 10 candidates)
  2. Cross-encoder re-ranking (top 3)
  3. Context building for answer generation

### **Evaluation Metrics**
- **ROUGE-1/2/L**: Lexical overlap measurement
- **BERTScore**: Semantic similarity using transformers
- **Both** for comprehensive evaluation

---

## 📁 File Structure

```
research-paper-summarizer/
├── app_new.py                 # 🌐 Streamlit web interface
├── ask_questions.py           # ❓ Interactive Q&A script
├── run_project.py             # Main pipeline batch executor
├── utils/
│   ├── pdf_extractor.py       # Extract text from PDF
│   ├── vector_store.py        # Embeddings & retrieval
│   ├── summarizer.py          # Local LLM integration (Ollama)
│   ├── evaluator.py           # ROUGE + BERTScore
│   ├── qa_engine.py           # ⭐ Smart Q&A engine
│   ├── rag_pipeline.py        # RAG orchestration
│   ├── hybrid_retriever.py    # Semantic + graph retrieval
│   └── ...
├── data/
│   ├── input/                 # 📥 PDFs to process
│   ├── output/                # 📤 Results & reports
│   └── summaries/             # Reference summaries
└── requirements.txt           # Python dependencies
```

---

## 🎮 Usage Examples

### Example 1: Batch Data Pipeline
```bash
python run_project.py
```

Output:
```
Processing 1 PDF file(s)...
[1/1] Processing: research_paper.pdf
  • Extracting text... ✓
  • Building metadata... ✓
  • Generating summary... ✓
  • Extracting knowledge triplets... ✓
```

### Example 2: Smart Q&A
```bash
python ask_questions.py --pdf "paper.pdf" --interactive
```

Interaction:
```
❓ Your question: What algorithm is proposed?
   🎯 Detected: ALGORITHM (confidence: 95%)
   🔄 Finding answer...
📝 The paper proposes [detailed answer]

❓ Your question: What dataset was used?
   🎯 Detected: DATASET (confidence: 88%)
   🔄 Finding answer...
📝 They evaluate on [dataset details]
```

### Example 3: Full Analysis Report
```bash
python ask_questions.py --pdf "paper.pdf" --report
```

Generates:
```
============================================================
📄 RESEARCH PAPER ANALYSIS REPORT
============================================================

📌 Title: [Paper Title]
📋 Abstract: [Abstract excerpt]
🏷️ Keywords: [keyword1, keyword2, ...]

❓ KEY QUESTIONS & ANSWERS:
1. OBJECTIVE QUESTION
   Q: What is the main objective?
   A: [Answer]

2. ALGORITHM QUESTION
   Q: What algorithm is proposed?
   A: [Answer]

... (and 6 more Q&A pairs)
```

### Example 4: Batch Processing with Q&A Reports
```bash
python ask_questions.py --dir "data/input" --output "qa_reports.json"
```

Processes all PDFs and saves:
```json
{
  "paper1.pdf": {
    "paper_info": {...},
    "qa_pairs": [...],
    "summary": "..."
  },
  "paper2.pdf": {...}
}
```

### Example 5: Web Interface
```bash
streamlit run app_new.py
```

Features in web interface:
- 📁 Upload any PDF
- 📝 Auto-generate summary
- ❓ Ask custom questions
- 📊 Full paper analysis
- 🔍 Q&A reports
- 📥 Download results

---

## ⚙️ Installation

### Requirements
- Python 3.10+
- 4GB+ RAM (16GB recommended for faster models)
- CUDA (optional, for GPU support)

### Setup
```bash
# Clone/navigate to project
cd research-paper-summarizer

# Install dependencies
pip install -r requirements.txt

# Download models (happens automatically on first run)
# Models will cache locally for faster future runs
```

### Dependencies
```
pymupdf                 # PDF text extraction
transformers           # Tokenizers
torch                  # Deep learning
sentence-transformers  # Embeddings & cross-encoder
nltk                   # Text processing
bert-score             # Evaluation metric
rouge-score            # ROUGE evaluation
streamlit              # Web interface
neo4j                  # Knowledge graphs
spacy                  # NLP
```

---

## 🔧 Advanced Configuration

### Environment Variables
```bash
# Skip heavy model loading for faster testing
export FAST_MODE=1

# Use only CPU (no CUDA)
export CUDA_VISIBLE_DEVICES=""

# Reduce memory usage
export TOKENIZERS_PARALLELISM=false
```

### Performance Tuning

**For Speed** (lower accuracy):
```python
# Use smaller models
model = SentenceTransformer("all-MiniLM-L6-v2")
# Reduce beam search: num_beams=1
```

**For Accuracy** (slower):
```python
# Use larger models
model = SentenceTransformer("all-t5-xxl-v1")  
# Increase beam search: num_beams=8
```

---

## 📊 Evaluation Results

### Test Results (Attention is All You Need paper)

| Metric | Score | Interpretation |
|--------|-------|-----------------|
| **ROUGE-1** | 0.331 | 33.1% unigram overlap with reference |
| **ROUGE-2** | 0.248 | 24.8% bigram overlap |
| **BERTScore** | 0.814 | **81.4% semantic similarity** ⭐ |

**Insight**: High BERTScore despite moderate ROUGE indicates semantically accurate but paraphrased summaries

### Generated Summary Example
> "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms."

✅ **Key aspects captured**:
- Architecture type (sequence transduction)
- Novel contribution (Transformer)
- Benefits (based on attention instead of recurrence)

---

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"
```bash
export CUDA_VISIBLE_DEVICES=""  # Use CPU instead
```

### Issue: "Model download too slow"
```bash
# Pre-download local LLM
ollama run llama3.2:1b
```

### Issue: PDF extraction fails
```
# Ensure PDF is not encrypted/scanned
# For scanned PDFs, use OCR first:
# pytesseract, paddleOCR, etc.
```

### Issue: Poor Q&A answers
```
# The paper might be too short or in unexpected format
# Adjust chunking parameters in vector_store.py
chunk_size = 1000  # Increase to capture more context
```

---

## 🎓 For Researchers

This system is perfect for:
- ✅ Literature reviews
- ✅ Paper summarization
- ✅ Quick information extraction
- ✅ Comparative analysis
- ✅ Metadata extraction
- ✅ Research paper cataloging

### Limitations
- Works best on **English** papers
- May struggle with **very short** papers (<500 words)
- **Scanned/image** PDFs need OCR preprocessing
- Performance depends on **paper structure** clarity

---

## 🚀 Future Enhancements

- [ ] Multi-language support
- [ ] Named entity linking to knowledge bases
- [ ] Figure and table extraction
- [ ] Citation analysis
- [ ] Related papers recommendation
- [ ] Academic metadata integration
- [ ] GPU optimization for large batches
- [ ] Export to BibTeX/RIS formats

---

## 📄 License

This project uses open-source models and libraries. See individual model licenses:
- Ollama / Llama 3.2: Meta AI
- Sentence-Transformers: Hugging Face (Apache 2.0)
- Cross-Encoder: Hugging Face (Apache 2.0)

---

## 🤝 Contributing

Found an issue or want to improve?
- Report bugs
- Suggest features
- Contribute code
- Improve documentation

---

## 📞 Support

For help:
1. Check this README
2. Review code comments
3. Check utils/qa_engine.py for question types
4. Run with `--help` flag for all options

---

**Made with ❤️ for researchers** | v3.0 with SmartQA
