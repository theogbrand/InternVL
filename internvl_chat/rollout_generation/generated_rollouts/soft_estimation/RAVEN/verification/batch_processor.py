import os
import json
import time
import datetime
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from openai import AzureOpenAI, BadRequestError
import uuid

@dataclass
class BatchJob:
    input_file: str
    status: str = "pending"  # pending, processing, completed, failed
    created_at: Optional[int] = None
    retries: int = 0
    file_id: Optional[str] = None
    batch_id: Optional[str] = None
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None

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
        
        # Track all jobs for this processor instance only
        self.completed_jobs: List[BatchJob] = []
        self.failed_jobs: List[BatchJob] = []
        self.my_active_jobs: List[BatchJob] = []  # Track only jobs submitted by THIS processor instance
        
        # Generate unique processor ID for safety
        self.processor_id = str(uuid.uuid4())[:8]
        
        # Log initialization
        self.logger.info(f"BatchProcessor initialized:")
        self.logger.info(f"  - Processor ID: {self.processor_id}")
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
    
    def upload_file(self, job: BatchJob) -> bool:
        """Upload a JSONL file to Azure OpenAI for batch processing."""
        try:
            filename = Path(job.input_file).name
            self.logger.info(f"Uploading file: {filename}")
            print(f"📤 Uploading file: {filename}")
            
            with open(job.input_file, "rb") as f:
                file_response = self.client.files.create(
                    file=f,
                    purpose="batch",
                    extra_body={"expires_after": {"seconds": 1209600, "anchor": "created_at"}}  # 14 days
                )
            
            job.file_id = file_response.id
            job.status = "uploaded"
            
            self.logger.info(f"File uploaded successfully: {job.file_id}")
            print(f"✅ File uploaded: {job.file_id}")
            
            # Log file expiration
            if file_response.expires_at:
                expiration = datetime.datetime.fromtimestamp(file_response.expires_at)
                self.logger.info(f"File expires at: {expiration}")
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to upload file {filename}: {str(e)}"
            self.logger.error(error_msg)
            print(f"❌ {error_msg}")
            job.status = "upload_failed"
            return False
    
    def create_batch(self, job: BatchJob) -> bool:
        """Create a batch job with the uploaded file."""
        try:
            filename = Path(job.input_file).name
            self.logger.info(f"Creating batch for file: {filename}")
            print(f"🚀 Creating batch for file: {filename}")
            
            retries = 0
            delay = self.initial_delay
            
            while retries < self.max_retries:
                try:
                    batch_response = self.client.batches.create(
                        input_file_id=job.file_id,
                        endpoint="/chat/completions",
                        completion_window="24h",
                        extra_body={"output_expires_after": {"seconds": 1209600, "anchor": "created_at"}}  # 14 days
                    )
                    
                    job.batch_id = batch_response.id
                    job.status = "batch_created"
                    job.created_at = batch_response.created_at
                    
                    self.logger.info(f"Batch created successfully: {job.batch_id}")
                    print(f"✅ Batch created: {job.batch_id}")
                    return True
                    
                except BadRequestError as e:
                    error_message = str(e)
                    
                    if 'token_limit_exceeded' in error_message:
                        retries += 1
                        if retries >= self.max_retries:
                            self.logger.error(f"Maximum retries ({self.max_retries}) reached for token limit")
                            print(f"❌ Maximum retries ({self.max_retries}) reached. Giving up.")
                            break
                        
                        self.logger.warning(f"Token limit exceeded. Waiting {delay} seconds before retry {retries}/{self.max_retries}")
                        print(f"⏳ Token limit exceeded. Waiting {delay} seconds before retry {retries}/{self.max_retries}...")
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                    else:
                        # Different error, raise immediately
                        raise e
            
            job.status = "batch_creation_failed"
            return False
            
        except Exception as e:
            error_msg = f"Failed to create batch for {filename}: {str(e)}"
            self.logger.error(error_msg)
            print(f"❌ {error_msg}")
            job.status = "batch_creation_failed"
            return False
    
    def monitor_batch(self, job: BatchJob) -> bool:
        """Monitor batch job until completion."""
        try:
            filename = Path(job.input_file).name
            self.logger.info(f"Monitoring batch {job.batch_id} for file: {filename}")
            print(f"⏳ Monitoring batch for file: {filename}")
            
            status = "validating"
            while status not in ("completed", "failed", "canceled"):
                time.sleep(60)  # Check every minute
                
                batch_response = self.client.batches.retrieve(job.batch_id)
                status = batch_response.status
                
                self.logger.info(f"Batch {job.batch_id} status: {status}")
                print(f"📊 Batch status: {status}")
                
                job.status = f"batch_{status}"
            
            if status == "completed":
                job.output_file_id = batch_response.output_file_id
                job.error_file_id = batch_response.error_file_id
                job.status = "completed"
                
                self.logger.info(f"Batch completed successfully: {job.batch_id}")
                print(f"✅ Batch completed successfully")
                return True
            else:
                job.status = "failed"
                if batch_response.errors:
                    for error in batch_response.errors.data:
                        self.logger.error(f"Batch error - Code: {error.code}, Message: {error.message}")
                        print(f"❌ Batch error - Code: {error.code}, Message: {error.message}")
                return False
                
        except Exception as e:
            error_msg = f"Failed to monitor batch {job.batch_id}: {str(e)}"
            self.logger.error(error_msg)
            print(f"❌ {error_msg}")
            job.status = "monitoring_failed"
            return False
    
    def process_results(self, job: BatchJob) -> bool:
        """Process and save batch results."""
        try:
            filename = Path(job.input_file).name
            output_filename = Path(job.input_file).stem + "_verification_results.json"
            output_path = self.verification_batches_dir / "verification_pipeline_outputs" / output_filename
            
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Processing results for file: {filename}")
            print(f"📝 Processing results for file: {filename}")
            
            verification_results = []
            error_sample_ids = []
            
            # Use output_file_id if available, otherwise error_file_id
            file_id = job.output_file_id or job.error_file_id
            
            if file_id:
                file_response = self.client.files.content(file_id)
                raw_responses = file_response.text.strip().split('\n')
                
                for raw_response in raw_responses:
                    json_response = json.loads(raw_response)
                    
                    # Check for error status codes
                    if json_response["response"]["status_code"] != 200:
                        error_sample_ids.append(json_response["custom_id"])
                        continue
                    
                    # Create verification entry for successful responses
                    verification_entry = {
                        "custom_id": json_response["custom_id"],
                        "verification_response": json_response["response"]["body"]["choices"][0]["message"]["content"]
                    }
                    
                    verification_results.append(verification_entry)
                
                # Save verification results
                with open(output_path, 'w') as f:
                    json.dump(verification_results, f, indent=2)
                
                self.logger.info(f"Saved {len(verification_results)} verification results to {output_path}")
                print(f"💾 Saved {len(verification_results)} verification results to {output_filename}")
                
                if error_sample_ids:
                    self.logger.warning(f"Error sample IDs with status_code != 200: {error_sample_ids}")
                    print(f"⚠️  {len(error_sample_ids)} samples had errors")
                else:
                    self.logger.info("All samples processed successfully with status_code 200")
                    print("✅ All samples processed successfully")
                
                return True
            else:
                self.logger.error("No output or error file ID available")
                print("❌ No output or error file ID available")
                return False
                
        except Exception as e:
            error_msg = f"Failed to process results for {filename}: {str(e)}"
            self.logger.error(error_msg)
            print(f"❌ {error_msg}")
            return False
    
    def process_single_job(self, job: BatchJob) -> bool:
        """Process a single batch job through the complete pipeline."""
        filename = Path(job.input_file).name
        
        # Step 1: Upload file
        if not self.upload_file(job):
            return False
        
        # Step 2: Create batch
        if not self.create_batch(job):
            return False
        
        # Step 3: Monitor batch
        if not self.monitor_batch(job):
            return False
        
        # Step 4: Process results
        if not self.process_results(job):
            return False
        
        self.logger.info(f"Successfully completed all steps for: {filename}")
        print(f"🎉 Successfully completed all steps for: {filename}")
        return True

    def check_job_status(self, job: BatchJob) -> str:
        """Check the status of a processing job."""
        return job.status
    
    def submit_single_batch(self, input_file: str) -> BatchJob:
        """Submit a single batch job."""
        job = BatchJob(input_file=input_file)
        filename = Path(input_file).name
        
        self.logger.info(f"Submitting batch job: {filename}")
        print(f"📤 Submitting batch job: {filename}")
        
        # Step 1: Upload file
        if not self.upload_file(job):
            job.status = "upload_failed"
            return job
            
        # Step 2: Create batch
        if not self.create_batch(job):
            job.status = "batch_creation_failed"
            return job
        
        job.status = "in_progress"
        self.my_active_jobs.append(job)  # Track as MY active job
        self.logger.info(f"[{self.processor_id}] Successfully submitted batch: {job.batch_id}")
        print(f"✅ Batch submitted: {job.batch_id}")
        return job
    
    def check_batch_completion(self, job: BatchJob) -> bool:
        """Check if a batch job is completed. Returns True if completed."""
        try:
            batch_response = self.client.batches.retrieve(job.batch_id)
            status = batch_response.status
            job.status = f"batch_{status}"
            
            filename = Path(job.input_file).name
            
            if status == "completed":
                job.output_file_id = batch_response.output_file_id
                job.error_file_id = batch_response.error_file_id
                job.status = "completed"
                
                self.logger.info(f"Batch completed: {filename}")
                print(f"✅ Batch completed: {filename}")
                
                # Process results immediately
                if self.process_results(job):
                    self.completed_jobs.append(job)
                    if job in self.my_active_jobs:
                        self.my_active_jobs.remove(job)  # Remove from my active jobs
                    self.logger.info(f"Results processed successfully: {filename}")
                    print(f"💾 Results processed: {filename}")
                else:
                    self.failed_jobs.append(job)
                    if job in self.my_active_jobs:
                        self.my_active_jobs.remove(job)  # Remove from my active jobs
                    self.logger.error(f"Result processing failed: {filename}")
                    print(f"❌ Result processing failed: {filename}")
                
                return True
                
            elif status in ("failed", "canceled"):
                job.status = "failed"
                if batch_response.errors:
                    for error in batch_response.errors.data:
                        self.logger.error(f"Batch {filename} error - Code: {error.code}, Message: {error.message}")
                
                self.failed_jobs.append(job)
                if job in self.my_active_jobs:
                    self.my_active_jobs.remove(job)  # Remove from my active jobs
                self.logger.error(f"Batch failed: {filename}")
                print(f"❌ Batch failed: {filename}")
                return True
                
            else:
                # Still in progress
                self.logger.info(f"Batch {filename} status: {status}")
                print(f"⏳ Batch {filename}: {status}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking batch {job.batch_id}: {str(e)}")
            print(f"⚠️  Error checking batch: {str(e)}")
            return False

    def process_all_batches(self, check_interval_minutes: int = 1):
        """Process batches one at a time: submit -> wait for completion -> submit next."""
        start_time = datetime.datetime.now()
        self.logger.info("="*60)
        self.logger.info("STARTING SEQUENTIAL BATCH PROCESSING")
        self.logger.info("="*60)
        
        input_files = self.discover_input_files()
        total_files = len(input_files)
        
        self.logger.info(f"Found {total_files} files to process sequentially")
        for i, input_file in enumerate(input_files, 1):
            self.logger.info(f"  {i:2d}. {Path(input_file).name}")
        
        print(f"🎯 Processing {total_files} batch jobs sequentially")
        
        try:
            current_job = None
            file_index = 0
            check_interval = check_interval_minutes * 60  # Convert to seconds
            
            while file_index < total_files or current_job:
                # If no current job and files remaining, submit next batch
                if not current_job and file_index < total_files:
                    input_file = input_files[file_index]
                    filename = Path(input_file).name
                    
                    self.logger.info(f"Processing file {file_index + 1}/{total_files}: {filename}")
                    print(f"\n📋 Processing file {file_index + 1}/{total_files}: {filename}")
                    
                    current_job = self.submit_single_batch(input_file)
                    file_index += 1
                    
                    # If submission failed, move to next file
                    if current_job.status in ["upload_failed", "batch_creation_failed"]:
                        self.failed_jobs.append(current_job)
                        current_job = None
                        continue
                
                # If we have a current job, check if it's completed
                if current_job:
                    if self.check_batch_completion(current_job):
                        # Job completed (successfully or failed), clear current job
                        current_job = None
                    else:
                        # Job still in progress, wait before next check
                        self.logger.info(f"Waiting {check_interval_minutes} minute(s) before next check...")
                        print(f"⏱️  Waiting {check_interval_minutes} minute(s) before next check...")
                        time.sleep(check_interval)
            
        except KeyboardInterrupt:
            self.logger.warning(f"[{self.processor_id}] Processing interrupted by user")
            print("\n🛑 Processing interrupted by user")
            self.cancel_my_active_batches()  # Cancel only MY active batches on interrupt
        except Exception as e:
            self.logger.error(f"[{self.processor_id}] Fatal error during processing: {str(e)}")
            print(f"❌ Fatal error: {str(e)}")
            self.cancel_my_active_batches()  # Cancel only MY active batches on error
        
        # Final summary
        end_time = datetime.datetime.now()
        total_runtime = end_time - start_time
        
        self.logger.info("="*60)
        self.logger.info("SEQUENTIAL PROCESSING COMPLETE")
        self.logger.info("="*60)
        self.logger.info(f"Session runtime: {total_runtime}")
        self.logger.info(f"Total files: {total_files}")
        self.logger.info(f"Successfully completed: {len(self.completed_jobs)}")
        self.logger.info(f"Failed jobs: {len(self.failed_jobs)}")
        
        print("\n" + "="*50)
        print("🏁 SEQUENTIAL PROCESSING COMPLETE")
        print("="*50)
        print(f"📊 Total files: {total_files}")
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
        self.logger.info("SEQUENTIAL PROCESSING LOG COMPLETE")
        self.logger.info("="*60)

    def cancel_my_active_batches(self) -> None:
        """Cancel only the batch jobs submitted by THIS processor instance."""
        if not self.my_active_jobs:
            self.logger.info(f"[{self.processor_id}] No active batch jobs to cancel for this processor")
            print("ℹ️  No active batch jobs to cancel for this processor")
            return
        
        self.logger.info(f"[{self.processor_id}] Cancelling {len(self.my_active_jobs)} batch jobs submitted by this processor...")
        print(f"🚫 Cancelling {len(self.my_active_jobs)} batch jobs submitted by this processor...")
        
        cancelled_count = 0
        failed_cancel_count = 0
        
        for job in self.my_active_jobs[:]:  # Create a copy to avoid modification during iteration
            if job.batch_id:
                try:
                    # Only cancel if this job was submitted by this processor instance
                    self.client.batches.cancel(job.batch_id)
                    job.status = "cancelled_by_user"
                    self.failed_jobs.append(job)
                    self.my_active_jobs.remove(job)
                    cancelled_count += 1
                    
                    filename = Path(job.input_file).name
                    self.logger.info(f"[{self.processor_id}] Cancelled batch: {job.batch_id} for file: {filename}")
                    print(f"🚫 Cancelled batch for file: {filename}")
                    
                except Exception as e:
                    failed_cancel_count += 1
                    self.logger.error(f"[{self.processor_id}] Failed to cancel batch {job.batch_id}: {str(e)}")
                    print(f"⚠️  Failed to cancel batch {job.batch_id}: {str(e)}")
            else:
                # Job doesn't have batch_id yet, just remove it
                self.my_active_jobs.remove(job)
                job.status = "cancelled_before_submission"
        
        self.logger.info(f"[{self.processor_id}] Cancellation summary: {cancelled_count} cancelled, {failed_cancel_count} failed")
        print(f"✅ Batch cancellation complete: {cancelled_count} cancelled, {failed_cancel_count} failed")
        
        if cancelled_count > 0:
            self.logger.info(f"[{self.processor_id}] Note: Only cancelled batches submitted by this processor instance")
            print("ℹ️  Note: Only cancelled batches submitted by this processor instance")

def main(check_interval_minutes: int = 1):
    """Main entry point."""
    processor = BatchProcessor(
        verification_batches_dir="verification_batches",
        max_retries=10,
        azure_endpoint="https://aisg-sj.openai.azure.com/",  # o4-mini endpoint
        api_key=os.getenv("AZURE_API_KEY")  # or provide directly
    )
    
    try:
        processor.process_all_batches(check_interval_minutes)
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