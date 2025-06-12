#!/usr/bin/env python3
"""
Test script for BatchProcessor
Tests the BatchProcessor by submitting the first 2 JSONL files from verification_batches
"""

import os
import sys
import signal
import argparse
from pathlib import Path

# Add the current directory to the path so we can import batch_processor
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from batch_processor import BatchProcessor

# Global processor variable for cleanup access
processor = None

def cleanup_handler(signum, frame):
    """Handle cleanup when script is interrupted."""
    print(f"\n🛑 Received signal {signum}. Cleaning up...")
    if processor:
        processor.cancel_my_active_batches()
    print("🧹 Cleanup completed")
    sys.exit(0)

def main():
    """Test the BatchProcessor with the first 2 JSONL files."""
    global processor
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test the BatchProcessor with the first 2 JSONL files.')
    parser.add_argument('--split', type=str, required=True, help='Split name (required parameter)')
    parser.add_argument('--azure-endpoint', type=str, default="https://aisg-sj.openai.azure.com/", 
                        help='Azure OpenAI endpoint URL')
    args = parser.parse_args()
    
    # Set up signal handlers for cleanup
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    
    # Path to the verification batches directory
    batches_dir = current_dir / "verification_pipeline_outputs" / "verification_batches"
    
    if not batches_dir.exists():
        print(f"❌ Batch directory not found: {batches_dir}")
        return
    
    # Get all JSONL files in the directory
    jsonl_files = sorted(list(batches_dir.glob("*.jsonl")))
    
    if len(jsonl_files) < 2:
        print(f"❌ Need at least 2 JSONL files, found {len(jsonl_files)}")
        return
    
    print(f"📁 Found {len(jsonl_files)} JSONL files")
    print(f"🎯 Testing with first 2 files:")
    for i, file in enumerate(jsonl_files[:2], 1):
        print(f"  {i}. {file.name}")
    
    # Create a temporary directory with just the first 2 files
    test_dir = current_dir / "test_verification_batches"
    test_dir.mkdir(exist_ok=True)
    
    try:
        # Copy first 2 files to test directory
        import shutil
        for file in jsonl_files[:2]:
            dest = test_dir / file.name
            if not dest.exists():
                shutil.copy2(file, dest)
                print(f"📋 Copied {file.name} to test directory")
        
        # Initialize BatchProcessor with test directory
        processor = BatchProcessor(
            verification_batches_dir=str(test_dir),
            max_retries=5,
            azure_endpoint=args.azure_endpoint,
            api_key=os.getenv("AZURE_API_KEY"),
            split=args.split
        )
        
        print(f"\n🚀 Starting batch processing test...")
        print(f"📂 Using test directory: {test_dir}")
        print(f"🎯 Using split: {args.split}")
        print(f"🔄 Max retries: 5")
        print(f"📋 Processing mode: Sequential (2 files)")
        print(f"🛑 Press Ctrl+C to stop and cancel only this processor's active batches")
        
        # Process the batches sequentially
        processor.process_all_batches(check_interval_minutes=1)
        
        print(f"\n✅ Test completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n🛑 Test interrupted by user")
        if processor:
            processor.cancel_my_active_batches()
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        if processor:
            processor.cancel_my_active_batches()
        import traceback
        traceback.print_exc()
    
    finally:
        # Final cleanup
        if processor:
            print("🧹 Final cleanup - cancelling any remaining batches from this processor...")
            processor.cancel_my_active_batches()
        
        # Keep test directory for inspection
        print(f"📂 Test directory preserved for inspection: {test_dir}")
        print(f"💡 You can find results and logs in: {test_dir}")

if __name__ == "__main__":
    main() 