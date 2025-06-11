import os
import json
import time
import datetime
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from openai import AzureOpenAI

@dataclass
class BatchJob:
    input_file: str
    status: str = "pending"  # pending, processing, completed, failed
    created_at: Optional[int] = None
    retries: int = 0

class BatchProcessor:
    def __init__(self, verification_batches_dir: str = "verification_batches", 
                 max_retries: int = 10, azure_endpoint: str = None, api_key: str = None):
        # Set up logging
        self._setup_logging()
        
        # Use provided parameters or fall back to defaults/environment
        endpoint = azure_endpoint or "https://aisg-sj.openai.azure.com/"
        key = api_key or os.getenv("AZURE_API_KEY")
        
        if not key:
            raise ValueError("API key must be provided either as parameter or AZURE_API_KEY environment variable")
        
        self.client = AzureOpenAI(
            api_key=key,  
            api_version="2025-03-01-preview",
            azure_endpoint=endpoint
        )
        
        self.verification_batches_dir = Path(verification_batches_dir)
        self.max_retries = max_retries
        self.initial_delay = 5
        
        # Track all jobs
        self.completed_jobs: List[BatchJob] = []
        self.failed_jobs: List[BatchJob] = []
        
        # Log initialization
        self.logger.info(f"BatchProcessor initialized:")
        self.logger.info(f"  - Endpoint: {endpoint}")
        self.logger.info(f"  - Batches directory: {self.verification_batches_dir}")
        self.logger.info(f"  - Max retries: {max_retries}")
        self.logger.info(f"  - Processing mode: Sequential")
    
    def _setup_logging(self):
        """Set up logging for the batch processor."""
        # Create logs directory if it doesn't exist
        log_dir = Path("batch_logs")
        log_dir.mkdir(exist_ok=True)
        
        # Generate log filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"batch_processor_{timestamp}.log"
        
        # Configure logging
        self.logger = logging.getLogger(f"BatchProcessor_{timestamp}")
        self.logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        print(f"📝 Logging to: {log_file}")
        print(f"📝 Monitor progress with: tail -f {log_file}")
        self.logger.info("="*60)
        self.logger.info("BATCH PROCESSOR SESSION STARTED")
        self.logger.info("="*60)
        
    def discover_input_files(self) -> List[str]:
        """Discover all JSONL files in the verification_batches directory."""
        self.logger.info(f"Discovering input files in: {self.verification_batches_dir}")
        
        if not self.verification_batches_dir.exists():
            error_msg = f"Directory {self.verification_batches_dir} not found"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        jsonl_files = list(self.verification_batches_dir.glob("*.jsonl"))
        if not jsonl_files:
            error_msg = f"No JSONL files found in {self.verification_batches_dir}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Log each file found
        self.logger.info(f"Found {len(jsonl_files)} JSONL files to process:")
        for i, file in enumerate(sorted(jsonl_files), 1):
            file_size = file.stat().st_size / (1024*1024)  # Size in MB
            self.logger.info(f"  {i:2d}. {file.name} ({file_size:.1f} MB)")
        
        print(f"📁 Found {len(jsonl_files)} JSONL files to process")
        return [str(f) for f in sorted(jsonl_files)]
    
    def check_job_status(self, job: BatchJob) -> str:
        """Check the status of a processing job."""
        # For direct processing, status is managed by the processing method
        return job.status
    
    def process_all_batches(self):
        """Main processing loop to handle all batch jobs."""
        start_time = datetime.datetime.now()
        self.logger.info("="*60)
        self.logger.info("STARTING BATCH PROCESSING SESSION")
        self.logger.info("="*60)
        
        # Discover input files in directory
        input_files = self.discover_input_files()
        
        self.logger.info(f"Processing session started with {len(input_files)} batch jobs:")
        for i, input_file in enumerate(input_files, 1):
            self.logger.info(f"  {i:2d}. {Path(input_file).name}")
        
        print(f"🎯 Starting sequential processing of {len(input_files)} batch jobs")
        
        # Process each file sequentially
        for i, input_file in enumerate(input_files, 1):
            job = BatchJob(input_file=input_file)
            filename = Path(input_file).name
            
            self.logger.info(f"Processing job {i}/{len(input_files)}: {filename}")
            print(f"\n📄 Processing job {i}/{len(input_files)}: {filename}")


        
        # Final summary
        end_time = datetime.datetime.now()
        total_runtime = end_time - start_time
        
        self.logger.info("="*60)
        self.logger.info("PROCESSING SESSION COMPLETE")
        self.logger.info("="*60)
        self.logger.info(f"Session runtime: {total_runtime}")
        self.logger.info(f"Total files processed: {len(input_files)}")
        self.logger.info(f"Completed jobs: {len(self.completed_jobs)}")
        self.logger.info(f"Failed jobs: {len(self.failed_jobs)}")
        
        print("\n" + "="*50)
        print("🏁 PROCESSING COMPLETE")
        print("="*50)
        print(f"✅ Completed: {len(self.completed_jobs)}")
        print(f"❌ Failed: {len(self.failed_jobs)}")
        print(f"⏱️  Total runtime: {total_runtime}")
        
        if self.failed_jobs:
            self.logger.error("Failed jobs:")
            print("\n💥 Failed jobs:")
            for job in self.failed_jobs:
                filename = Path(job.input_file).name
                self.logger.error(f"  - {filename} (status: {job.status})")
                print(f"  - {filename} (status: {job.status})")
        
        if self.completed_jobs:
            self.logger.info("Successfully processed files:")
            print("\n🎉 Successfully processed files:")
            for job in self.completed_jobs:
                filename = Path(job.input_file).name
                self.logger.info(f"  - {filename}")
                print(f"  - {filename}")
        
        self.logger.info("="*60)
        self.logger.info("SESSION LOG COMPLETE")
        self.logger.info("="*60)

def main():
    """Main entry point."""
    processor = BatchProcessor(
        verification_batches_dir="verification_batches",
        max_retries=10,
        azure_endpoint="https://aisg-sj.openai.azure.com/",  # o4-mini endpoint
        api_key=os.getenv("AZURE_API_KEY")  # or provide directly
    )
    
    try:
        processor.process_all_batches()
    except KeyboardInterrupt:
        print("\n🛑 Processing interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")

    
def run_processor(deployment_config):
    """Run a single processor for a deployment."""
    try:
        print(f"🚀 Starting processor for {deployment_config['name']}")
        processor = BatchProcessor(
            verification_batches_dir=f"verification_batches_{deployment_config['name']}",
            max_retries=10,
            azure_endpoint=deployment_config['endpoint'],
            api_key=deployment_config['api_key']
        )
        processor.process_all_batches()
        print(f"✅ Processor for {deployment_config['name']} completed")
    except Exception as e:
        print(f"❌ Processor for {deployment_config['name']} failed: {str(e)}")

if __name__ == "__main__":
    main() 