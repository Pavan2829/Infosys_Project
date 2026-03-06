"""
Comprehensive Project Executor
This script runs the entire research paper summarizer pipeline
"""

import os
import sys
import json
from datetime import datetime

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)

def check_dependencies():
    """Check if all required packages are installed"""
    print_header("Checking Dependencies")
    
    required_packages = {
        'pymupdf': 'fitz',
        'transformers': 'transformers',
        'torch': 'torch',
        'streamlit': 'streamlit',
        'neo4j': 'neo4j',
        'sentence_transformers': 'sentence_transformers',
        'numpy': 'numpy',
        'spacy': 'spacy',
        'rouge-score': 'rouge_score'
    }
    
    missing = []
    for package, module_name in required_packages.items():
        try:
            __import__(module_name)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        return False
    
    print("\n✓ All dependencies are installed!")
    return True

def check_project_structure():
    """Verify project directory structure"""
    print_header("Checking Project Structure")
    
    required_dirs = [
        'data/input',
        'data/output',
        'data/summaries',
        'data/kg',
        'utils'
    ]
    
    required_files = [
        'run_project.py',
        'app_new.py',
        'requirements.txt',
        'utils/pdf_extractor.py',
        'utils/metadata_builder.py',
        'utils/summarizer.py',
        'utils/triplet_extractor.py',
        'utils/neo4j_loader.py',
        'utils/vector_store.py'
    ]
    
    all_exist = True
    
    for dir_path in required_dirs:
        full_path = os.path.join(os.getcwd(), dir_path)
        if os.path.exists(full_path):
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ (Missing)")
            all_exist = False
    
    for file_path in required_files:
        full_path = os.path.join(os.getcwd(), file_path)
        if os.path.exists(full_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (Missing)")
            all_exist = False
    
    if all_exist:
        print("\n✓ Project structure is complete!")
    else:
        print("\n⚠️  Some files or directories are missing!")
    
    return all_exist

def check_input_files():
    """Check for PDF files in data/input"""
    print_header("Checking Input Files")
    
    input_dir = os.path.join(os.getcwd(), 'data', 'input')
    
    if not os.path.exists(input_dir):
        print(f"✗ Input directory not found: {input_dir}")
        return False
    
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("⚠️  No PDF files found in data/input/")
        print("\nTo run the pipeline, please add PDF files to:")
        print(f"  {input_dir}/")
        return False
    
    print(f"Found {len(pdf_files)} PDF file(s):")
    for pdf in pdf_files[:10]:  # Show first 10
        print(f"  • {pdf}")
    
    if len(pdf_files) > 10:
        print(f"  ... and {len(pdf_files) - 10} more")
    
    print(f"\n✓ Ready to process!")
    return True

def run_pipeline():
    """Run the main pipeline"""
    print_header("Running PDF Processing Pipeline")
    
    try:
        # Import here to catch import errors
        from utils.pdf_extractor import extract_text_from_pdf
        from utils.metadata_builder import build_metadata
        from utils.summarizer import generate_summary
        from utils.triplet_extractor import extract_triplets
        from utils.neo4j_loader import Neo4jLoader
        import fitz
        
        PDF_FOLDER = os.path.join(os.getcwd(), "data", "input")
        METADATA_PATH = os.path.join(os.getcwd(), "data", "output", "metadata.json")
        SUMMARY_PATH = os.path.join(os.getcwd(), "data", "summaries", "summary.txt")
        ENTITIES_PATH = os.path.join(os.getcwd(), "data", "kg", "entities.json")
        TRIPLETS_PATH = os.path.join(os.getcwd(), "data", "kg", "triplets.json")
        
        pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
        
        if not pdf_files:
            print("✗ No PDF files found in data/input/")
            return False
        
        print(f"\nProcessing {len(pdf_files)} PDF file(s)...\n")
        
        all_metadata = []
        all_summaries = []
        
        try:
            loader = Neo4jLoader()
            neo4j_available = True
            print("✓ Neo4j connection established")
        except Exception as e:
            neo4j_available = False
            print(f"⚠️  Neo4j not available: {str(e)}")
        
        for idx, pdf_name in enumerate(pdf_files, 1):
            PDF_PATH = os.path.join(PDF_FOLDER, pdf_name)
            print(f"\n[{idx}/{len(pdf_files)}] Processing: {pdf_name}")
            
            try:
                # Extract text
                print("  • Extracting text...", end=" ")
                text = extract_text_from_pdf(PDF_PATH)
                print("✓")
                
                # Get file info
                doc = fitz.open(PDF_PATH)
                pages = len(doc)
                doc.close()
                text_length = len(text)
                
                # Build metadata
                print("  • Building metadata...", end=" ")
                metadata = build_metadata(text, os.path.basename(PDF_PATH), pages, text_length)
                all_metadata.append(metadata)
                print("✓")
                
                # Generate summary
                print("  • Generating summary...", end=" ")
                summary = generate_summary(text)
                all_summaries.append(f"\n--- Summary for {pdf_name} ---\n{summary}\n")
                print("✓")
                
                # Extract triplets
                print("  • Extracting knowledge triplets...", end=" ")
                triplets = extract_triplets(text)
                print(f"✓ ({len(triplets)} triplets)")
                
                # Load to Neo4j if available
                if neo4j_available:
                    print("  • Loading to Neo4j...", end=" ")
                    try:
                        loader.load_metadata(metadata)
                        for t in triplets:
                            loader.create_relationship(t[0], t[1], t[2])
                        print("✓")
                    except Exception as e:
                        print(f"⚠️  ({str(e)[:30]}...)")
                
            except Exception as e:
                print(f"\n✗ Error processing {pdf_name}: {str(e)}")
                continue
        
        # Save all metadata
        os.makedirs("data/output", exist_ok=True)
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, indent=4)
        
        # Save all summaries
        os.makedirs("data/summaries", exist_ok=True)
        with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("RESEARCH PAPER SUMMARY REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
            for summary in all_summaries:
                f.write(summary)
        
        if neo4j_available:
            loader.close()
        
        print(f"\n{'='*80}")
        print(f"✓ PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print(f"\nOutput files generated:")
        print(f"  • Metadata: {METADATA_PATH}")
        print(f"  • Summaries: {SUMMARY_PATH}")
        print(f"  • Entities: {ENTITIES_PATH}")
        print(f"  • Triplets: {TRIPLETS_PATH}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def show_results():
    """Display results summary"""
    print_header("Results Summary")
    
    output_dir = os.path.join(os.getcwd(), "data", "output")
    summaries_dir = os.path.join(os.getcwd(), "data", "summaries")
    kg_dir = os.path.join(os.getcwd(), "data", "kg")
    
    # Metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"✓ Processed {len(metadata)} document(s)")
        total_pages = sum(m.get('pages', 0) for m in metadata)
        total_text = sum(m.get('text_length', 0) for m in metadata)
        print(f"  • Total pages: {total_pages}")
        print(f"  • Total text length: {total_text:,} characters")
    
    # Summaries
    summary_path = os.path.join(summaries_dir, "summary.txt")
    if os.path.exists(summary_path):
        size = os.path.getsize(summary_path)
        print(f"\n✓ Summary report generated")
        print(f"  • File size: {size:,} bytes")
    
    # Knowledge Graph
    entities_path = os.path.join(kg_dir, "entities.json")
    triplets_path = os.path.join(kg_dir, "triplets.json")
    
    if os.path.exists(entities_path):
        with open(entities_path, 'r') as f:
            entities = json.load(f)
        print(f"\n✓ Knowledge Graph extracted")
        print(f"  • Entities: {len(entities)}")
    
    if os.path.exists(triplets_path):
        with open(triplets_path, 'r') as f:
            triplets = json.load(f)
        print(f"  • Triplets: {len(triplets)}")

def show_usage():
    """Show how to use the project"""
    print_header("How to Use Your Project")
    
    print("\n1. RUN THE PIPELINE (Batch Processing):")
    print("   python run_project.py")
    print("\n   This will:")
    print("   • Extract text from all PDFs in data/input/")
    print("   • Generate summaries")
    print("   • Extract entities and relationships")
    print("   • Create knowledge graph in Neo4j (if running)")
    
    print("\n2. RUN THE STREAMLIT APP (Interactive):")
    print("   streamlit run app_new.py")
    print("\n   This will:")
    print("   • Start a web interface")
    print("   • Allow uploading a single PDF")
    print("   • Ask questions about the paper")
    print("   • Get hybrid RAG answers using vector + knowledge graph")
    
    print("\n3. REQUIRED SETUP FOR APP:")
    print("   • You need a running Neo4j database")
    print("   • Update the connection details in utils/neo4j_loader.py if needed")
    print("   • Connection: bolt://localhost:7687")

def main():
    """Main execution function"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  RESEARCH PAPER SUMMARIZER - PROJECT EXECUTOR".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Check dependencies
    if not check_dependencies():
        print("\n⚠️  Please install missing dependencies:")
        print("   pip install -r requirements.txt")
        return
    
    # Check project structure
    if not check_project_structure():
        print("\n⚠️  Please fix the project structure!")
        return
    
    # Check input files
    if not check_input_files():
        print("\n⚠️  Please add PDF files to data/input/ and run again!")
        show_usage()
        return
    
    # Run pipeline
    if run_pipeline():
        show_results()
        show_usage()
    else:
        print("\n⚠️  Pipeline execution failed!")
        show_usage()

if __name__ == "__main__":
    main()
