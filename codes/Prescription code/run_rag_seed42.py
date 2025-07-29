#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run TCM RAG system with random seed 42
Generate result files with _42 suffix
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tcm_rag_processor import TCMRAGProcessor

def main():
    """Run RAG system with random seed 42"""
    print("TCM RAG System - Random Seed 42 Version")
    print("=" * 60)
    print("Setting random seed: 42")
    print("Generated file suffix: _42")
    print("=" * 60)
    
    # Check if data directory exists
    if not os.path.exists("data"):
        print("Error: data directory not found")
        print("Please ensure this script is run in a path containing the data directory")
        return
    
    # Check for txt files in data directory
    import glob
    txt_files = glob.glob("data/*.txt")
    if not txt_files:
        print("Error: No .txt files found in data directory")
        return
    
    print(f"Found {len(txt_files)} .txt files")
    print("Starting RAG system...")
    
    # Create processor with random seed 42
    processor = TCMRAGProcessor(random_seed=42)
    
    try:
        # Run complete processing pipeline with _42 suffix
        processor.run_complete_process(output_suffix="_new42new——2")
        
        print("\n" + "=" * 60)
        print("Processing complete! Generated files:")
        if os.path.exists("zhenduan_42_new.json"):
            print("✓ zhenduan_42_new.json - Diagnosis case data")
        if os.path.exists("gold_standard_42_new.json"):
            print("✓ gold_standard_42_new.json - Gold standard case data")
        if os.path.exists("rag_output_42_new.json"):
            print("✓ rag_output_42.json - RAG system processing results")
        if os.path.exists("rag_full_responses_42.json"):
            print("✓ rag_full_responses_42.json - Complete response content")
        if os.path.exists("rag_output_improved_42.json"):
            print("✓ rag_output_improved_42.json - Improved final results")
            
        print("\n" + "=" * 60)
        print("Random seed 42 version features:")
        print("- Random seed: 42, ensuring reproducible results")
        print("- File suffix: _42, for easy version distinction")
        print("- Processing logic: Same as standard version")
        print("=" * 60)
            
    except KeyboardInterrupt:
        print("\n\nUser interrupted processing")
        print("Current progress saved, can restart program to continue processing")
    except Exception as e:
        print(f"\nError occurred during processing: {e}")
        print("Please check error information and retry")
        import traceback
        print(f"Detailed error information:\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()
