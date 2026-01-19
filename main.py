"""
ArXiv Similarity Search - Main Entry Point (MOCK VERSION)
Interactive CLI for finding and analyzing similar research papers

THIS IS A MOCK VERSION - NO GPU/LLM REQUIRED
Returns realistic dummy data for frontend development

Usage:
    python main.py

Features:
    - TWO MODES: ArXiv (online) or Local Database (offline)
    - Mock keyword extraction (instant)
    - Mock paper search (instant)
    - Mock embeddings + FAISS (instant)
    - Mock reranking (instant)
    - Mock comparative analysis (instant)
    - Mock paper summaries (instant)
"""

import os
import sys
import json
import yaml
import shutil
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline_orchestrator import MockPipelineOrchestrator


def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        'data/temp_pdfs',
        'data/sample_pdfs',
        'data/local_database',
        'data/faiss_indices',
        'outputs',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        # Create .gitkeep files
        gitkeep_path = os.path.join(directory, '.gitkeep')
        if not os.path.exists(gitkeep_path):
            Path(gitkeep_path).touch()


def cleanup_temp_files():
    """Clean up temporary PDF files"""
    temp_dir = 'data/temp_pdfs'
    if os.path.exists(temp_dir):
        try:
            for file in os.listdir(temp_dir):
                if file.endswith('.pdf'):
                    file_path = os.path.join(temp_dir, file)
                    os.remove(file_path)
            print(f"[OK] Cleaned up temporary PDFs from {temp_dir}/")
        except Exception as e:
            print(f"[WARNING] Error cleaning temp files: {e}")


def display_banner():
    """Display welcome banner"""
    banner = """
================================================================================
                                                                              
         ARXIV SIMILARITY SEARCH SYSTEM (MOCK VERSION - NO GPU)
                                                                              
          AI-Powered Research Paper Discovery & Analysis Engine               
                                                                              
  - TWO MODES: ArXiv (online) & Local Database (offline)                     
  - Mock Semantic Search (instant results)                                   
  - Mock Intelligent Reranking (instant results)                             
  - Mock Automated Analysis & Summarization (instant results)                
                                                                              
  ⚠️  THIS IS A MOCK VERSION FOR FRONTEND DEVELOPMENT                        
  ⚠️  NO GPU/LLM REQUIRED - RETURNS REALISTIC DUMMY DATA                     
                                                                              
================================================================================
"""
    print(banner)


def display_mode_selection(config: dict) -> str:
    """
    Display mode selection menu and get user choice
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Selected mode: "arxiv" or "local"
    """
    print(f"\n{'='*80}")
    print("SELECT SEARCH MODE")
    print(f"{'='*80}\n")
    
    # Check for PDFs
    local_db_path = config.get('local_database', {}).get('folder_path', 'data/local_database')
    sample_pdfs_path = config.get('paths', {}).get('sample_pdfs', 'data/sample_pdfs')
    
    local_pdfs = []
    if os.path.exists(local_db_path):
        local_pdfs = [f for f in os.listdir(local_db_path) if f.endswith('.pdf')]
    
    sample_pdfs = []
    if os.path.exists(sample_pdfs_path):
        sample_pdfs = [f for f in os.listdir(sample_pdfs_path) if f.endswith('.pdf')]
    
    pdf_count = max(len(local_pdfs), len(sample_pdfs))
    is_valid = pdf_count > 0
    
    print("Available Modes:\n")
    print("  1. ArXiv Mode      - Mock search from ArXiv (instant)")
    print("  2. Local Database  - Mock search from local papers (instant)")
    
    print(f"\n{'-'*80}")
    print("Local Database Status:")
    print(f"  Location: {local_db_path}")
    print(f"  Sample PDFs: {sample_pdfs_path}")
    
    if is_valid:
        print(f"  Status: ✓ Ready ({pdf_count} PDF files found)")
        print(f"  Index: ✓ Mock (instant loading)")
    else:
        print(f"  Status: ⚠️  No PDF files found")
        print(f"  Note: Add PDF files to {sample_pdfs_path}/ for realistic filenames")
        print(f"  Note: Mock will work anyway with generated filenames")
    
    print(f"{'-'*80}\n")
    
    while True:
        choice = input("Enter mode (1 for ArXiv, 2 for Local): ").strip()
        
        if choice == '1':
            print("\n[OK] ArXiv Mode selected (Mock)")
            return "arxiv"
        elif choice == '2':
            print("\n[OK] Local Database Mode selected (Mock)")
            return "local"
        else:
            print("[WARNING] Invalid choice. Please enter 1 or 2.")
            continue


def display_help(mode: str):
    """Display help information based on mode"""
    
    if mode == "arxiv":
        help_text = """
================================================================================
                   COMMAND REFERENCE - ARXIV MODE (MOCK)                       
================================================================================

MAIN COMMANDS:
  - Enter abstract       : Start new search with research abstract
  - 1-5                 : Generate summary for paper (after search results)
  - next                : Start new search
  - switch              : Switch to Local Database mode
  - help                : Show this help message
  - stats               : Show session statistics
  - exit / quit         : Exit the application

WORKFLOW:
  1. Paste your research abstract (press Enter twice to finish)
  2. System generates mock keywords (instant)
  3. System generates mock ArXiv papers (instant)
  4. System generates Top 10 mock similar papers (instant)
  5. Displays Top 5 mock reranked papers with mock analysis (instant)
  6. Enter paper number (1-5) to generate mock detailed summary (instant)
  7. Type 'next' for new search or 'exit' to quit

MOCK MODE NOTES:
  - No actual ArXiv API calls
  - No PDF downloads needed
  - No GPU/LLM processing
  - All results are realistic dummy data
  - Instant results (simulated delays for realism)

TIPS:
  - Abstracts should be at least 50 characters
  - All results are mock data for frontend testing
  - Switch to production version when ready
"""
    else:  # local mode
        help_text = """
================================================================================
                 COMMAND REFERENCE - LOCAL DATABASE MODE (MOCK)                
================================================================================

MAIN COMMANDS:
  - Enter abstract       : Start new search with research abstract
  - 1-5                 : Generate summary for paper (after search results)
  - next                : Start new search
  - switch              : Switch to ArXiv mode
  - help                : Show this help message
  - stats               : Show session statistics
  - exit / quit         : Exit the application

WORKFLOW:
  1. Paste your research abstract (press Enter twice to finish)
  2. System generates mock index (instant)
  3. System generates Top 10 mock similar papers (instant)
  4. Displays Top 5 mock reranked papers with mock analysis (instant)
  5. Enter paper number (1-5) to generate mock detailed summary (instant)
  6. Type 'next' for new search or 'exit' to quit

MOCK MODE NOTES:
  - No actual PDF processing
  - No actual embeddings computed
  - No GPU/LLM processing
  - All results are realistic dummy data
  - Instant results (simulated delays for realism)

TIPS:
  - Add your PDFs to data/sample_pdfs/ for realistic filenames
  - All results are mock data for frontend testing
  - Switch to production version when ready
"""
    
    print(help_text)


def get_multiline_input(prompt: str) -> str:
    """Get multi-line input from user"""
    print(prompt)
    print("(Press Enter twice to finish)\n")
    
    lines = []
    empty_count = 0
    
    while True:
        try:
            line = input()
            if not line.strip():
                empty_count += 1
                if empty_count >= 2:
                    break
            else:
                empty_count = 0
                lines.append(line)
        except EOFError:
            break
    
    return ' '.join(lines).strip()


def interactive_mode(orchestrator, config: dict):
    """Run interactive CLI mode"""
    
    session_stats = {
        'searches_performed': 0,
        'papers_analyzed': 0,
        'summaries_generated': 0,
        'mode_switches': 0,
        'start_time': datetime.now()
    }
    
    current_results = None
    current_mode = orchestrator.mode
    
    while True:
        try:
            print("\n" + "="*80)
            
            # Get user input
            if current_results is None:
                print(f"\n[Mode: {current_mode.upper()} - MOCK] Enter your research abstract (or 'help', 'switch', 'exit'):")
                user_input = get_multiline_input("")
            else:
                print("\nOptions:")
                print("  - Enter 1-5 to summarize that paper")
                print("  - Type 'next' for new search")
                print("  - Type 'switch' to change mode")
                print("  - Type 'help' for commands")
                print("  - Type 'exit' to quit")
                user_input = input("\n> ").strip()
            
            # Handle commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n" + "="*80)
                print("SESSION SUMMARY")
                print("="*80)
                print(f"Searches performed: {session_stats['searches_performed']}")
                print(f"Papers analyzed: {session_stats['papers_analyzed']}")
                print(f"Summaries generated: {session_stats['summaries_generated']}")
                print(f"Mode switches: {session_stats['mode_switches']}")
                duration = datetime.now() - session_stats['start_time']
                print(f"Session duration: {duration.seconds // 60} minutes")
                print("="*80)
                
                print("\n[OK] Thank you for using ArXiv Similarity Search (Mock)!")
                break
            
            elif user_input.lower() == 'help':
                display_help(current_mode)
                continue
            
            elif user_input.lower() == 'switch':
                # Switch mode
                new_mode = "local" if current_mode == "arxiv" else "arxiv"
                
                print(f"\n[OK] Switching from {current_mode.upper()} mode to {new_mode.upper()} mode...")
                
                # Reinitialize orchestrator with new mode
                orchestrator = MockPipelineOrchestrator(mode=new_mode)
                current_mode = new_mode
                current_results = None
                session_stats['mode_switches'] += 1
                
                print(f"[OK] Now in {new_mode.upper()} mode (Mock)")
                continue
            
            elif user_input.lower() == 'stats':
                print("\n" + "="*80)
                print("SESSION STATISTICS")
                print("="*80)
                print(f"Current Mode: {current_mode.upper()} (Mock)")
                print(f"Searches performed: {session_stats['searches_performed']}")
                print(f"Papers analyzed: {session_stats['papers_analyzed']}")
                print(f"Summaries generated: {session_stats['summaries_generated']}")
                print(f"Mode switches: {session_stats['mode_switches']}")
                duration = datetime.now() - session_stats['start_time']
                print(f"Session duration: {duration.seconds // 60} minutes")
                print("="*80)
                continue
            
            elif user_input.lower() == 'cleanup':
                cleanup_temp_files()
                continue
            
            elif user_input.lower() == 'next':
                current_results = None
                print("\n[OK] Starting new search...")
                continue
            
            # Handle paper number for summary
            elif current_results is not None and user_input.isdigit():
                paper_num = int(user_input)
                
                if 1 <= paper_num <= 5:
                    print(f"\n{'='*80}")
                    print(f"GENERATING MOCK SUMMARY FOR PAPER {paper_num}")
                    print(f"{'='*80}\n")
                    
                    summary = orchestrator.generate_summary(
                        current_results['top_5_papers'][paper_num - 1]
                    )
                    
                    if summary:
                        orchestrator.display_summary(summary)
                        session_stats['summaries_generated'] += 1
                        
                        # Add summary to results
                        current_results['top_5_papers'][paper_num - 1]['summary'] = summary
                        
                        # Save updated results
                        orchestrator.save_results(current_results)
                    else:
                        print("[WARNING] Failed to generate mock summary")
                else:
                    print(f"[WARNING] Invalid paper number. Please enter 1-5.")
                continue
            
            # Handle new search
            elif len(user_input) >= 50:
                print(f"\n{'='*80}")
                print("STARTING NEW MOCK SEARCH")
                print(f"{'='*80}\n")
                
                # Run full mock pipeline
                results = orchestrator.run_pipeline(user_input)
                
                if results and results.get('top_5_papers'):
                    current_results = results
                    session_stats['searches_performed'] += 1
                    session_stats['papers_analyzed'] += len(results['top_5_papers'])
                    
                    print(f"\n{'='*80}")
                    print("[OK] MOCK SEARCH COMPLETE - Top 5 Papers Ready")
                    print(f"{'='*80}")
                    print("\nYou can now:")
                    print("  - Enter 1-5 to generate mock detailed summary for that paper")
                    print("  - Type 'next' to start a new search")
                    print("  - Type 'switch' to change mode")
                    print("  - Type 'exit' to quit")
                else:
                    print("\n[WARNING] Mock search returned no results.")
                    current_results = None
            
            elif user_input and len(user_input) < 50:
                print("\n[WARNING] Abstract too short. Please provide at least 50 characters.")
                continue
            
            else:
                print("\n[WARNING] Invalid input. Type 'help' for available commands.")
                continue
        
        except KeyboardInterrupt:
            print("\n\n[WARNING] Interrupted by user")
            break
        
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
            print("\nYou can continue with a new search or type 'exit' to quit.")
            current_results = None


def main():
    """Main entry point"""
    
    # Display banner
    display_banner()
    
    # Create directories
    print("Initializing mock system...")
    create_directories()
    print("[OK] Directories created\n")
    
    # Check config file
    if not os.path.exists('config.yaml'):
        print("[ERROR] config.yaml not found!")
        print("Please create config.yaml.")
        print("Using default mock configuration...")
        config = {
            'arxiv': {'max_results': 20},
            'reranking': {'top_k': 5},
            'local_database': {'folder_path': 'data/local_database'},
            'paths': {'sample_pdfs': 'data/sample_pdfs'}
        }
    else:
        # Load config
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    
    print("Mock Configuration:")
    print(f"  Mode: MOCK (No GPU/LLM)")
    print(f"  ArXiv Max Results: {config.get('arxiv', {}).get('max_results', 20)}")
    print(f"  Top K Papers: {config.get('reranking', {}).get('top_k', 5)}")
    print(f"  Processing: Instant (simulated delays for realism)")
    
    # Mode selection
    selected_mode = display_mode_selection(config)
    
    # Initialize Mock Pipeline Orchestrator
    print("\n" + "="*80)
    print("INITIALIZING MOCK PIPELINE (NO GPU REQUIRED)")
    orchestrator = MockPipelineOrchestrator(mode=selected_mode)
    
    print(f"\n{'='*80}")
    print("[OK] MOCK SYSTEM READY")
    print(f"{'='*80}\n")
    
    # Display quick help
    if selected_mode == "arxiv":
        print("QUICK START (ArXiv Mode - MOCK):")
        print("  1. Paste your research abstract (press Enter twice)")
        print("  2. System generates mock results instantly")
        print("  3. Review Top 5 similar papers with mock analysis")
        print("  4. Enter 1-5 to get detailed mock summary")
        print("  5. Type 'help' for all commands\n")
    else:
        print("QUICK START (Local Database Mode - MOCK):")
        print("  1. Paste your research abstract (press Enter twice)")
        print("  2. System generates mock results instantly")
        print("  3. Review Top 5 similar papers with mock analysis")
        print("  4. Enter 1-5 to get detailed mock summary")
        print("  5. Type 'help' for all commands\n")
    
    print("⚠️  REMINDER: This is a MOCK version for frontend development")
    print("⚠️  All results are realistic dummy data (no actual AI processing)\n")
    
    # Start interactive mode
    try:
        interactive_mode(orchestrator, config)
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
