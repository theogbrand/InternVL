import os
import json
import time
import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from openai import AzureOpenAI, BadRequestError

@dataclass
class BatchJob:
    input_file: str
    file_id: Optional[str] = None
    batch_id: Optional[str] = None
    status: str = "pending"  # pending, uploading, submitted, running, completed, failed
    created_at: Optional[int] = None
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None
    retries: int = 0

class BatchProcessor:
    def __init__(self, verification_batches_dir: str = "verification_batches", 
                 max_concurrent_batches: int = 5, max_retries: int = 10,
                 azure_endpoint: str = None, api_key: str = None):
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
        self.max_concurrent_batches = max_concurrent_batches
        self.max_retries = max_retries
        self.initial_delay = 5
        
        # Track all jobs
        self.pending_jobs: List[BatchJob] = []
        self.active_jobs: Dict[str, BatchJob] = {}  # batch_id -> BatchJob
        self.completed_jobs: List[BatchJob] = []
        self.failed_jobs: List[BatchJob] = []
        
    def discover_input_files(self) -> List[str]:
        """Discover all JSONL files in the verification_batches directory."""
        if not self.verification_batches_dir.exists():
            raise FileNotFoundError(f"Directory {self.verification_batches_dir} not found")
        
        jsonl_files = list(self.verification_batches_dir.glob("*.jsonl"))
        if not jsonl_files:
            raise FileNotFoundError(f"No JSONL files found in {self.verification_batches_dir}")
        
        print(f"📁 Found {len(jsonl_files)} JSONL files to process")
        return [str(f) for f in jsonl_files]
    
    def upload_file_with_retry(self, job: BatchJob) -> bool:
        """Upload file with retry logic for token limits."""
        job.status = "uploading"
        retries = 0
        delay = self.initial_delay
        
        while retries < self.max_retries:
            try:
                print(f"📤 Uploading {job.input_file}...")
                file_response = self.client.files.create(
                    file=open(job.input_file, "rb"), 
                    purpose="batch",
                    extra_body={"expires_after": {"seconds": 1209600, "anchor": "created_at"}}
                )
                
                job.file_id = file_response.id
                print(f"✅ File uploaded successfully: {job.file_id}")
                return True
                
            except BadRequestError as e:
                error_message = str(e)
                if 'token_limit_exceeded' in error_message:
                    retries += 1
                    if retries >= self.max_retries:
                        print(f"❌ Upload failed after {self.max_retries} retries for {job.input_file}")
                        job.status = "failed"
                        return False
                    
                    print(f"⏳ Token limit exceeded during upload. Waiting {delay}s before retry {retries}/{self.max_retries}...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    print(f"❌ Upload error for {job.input_file}: {error_message}")
                    job.status = "failed"
                    return False
            except Exception as e:
                print(f"❌ Unexpected upload error for {job.input_file}: {str(e)}")
                job.status = "failed"
                return False
        
        return False
    
    def create_batch_with_retry(self, job: BatchJob) -> bool:
        """Create batch job with retry logic for token limits."""
        if not job.file_id:
            return False
        
        retries = 0
        delay = self.initial_delay
        
        while retries < self.max_retries:
            try:
                print(f"🚀 Creating batch job for {job.input_file}...")
                batch_response = self.client.batches.create(
                    input_file_id=job.file_id,
                    endpoint="/chat/completions",
                    completion_window="24h",
                    extra_body={"output_expires_after": {"seconds": 1209600, "anchor": "created_at"}}
                )
                
                job.batch_id = batch_response.id
                job.status = "submitted"
                job.created_at = batch_response.created_at
                
                print(f"✅ Batch created successfully: {job.batch_id}")
                return True
                
            except BadRequestError as e:
                error_message = str(e)
                if 'token_limit_exceeded' in error_message:
                    retries += 1
                    if retries >= self.max_retries:
                        print(f"❌ Batch creation failed after {self.max_retries} retries for {job.input_file}")
                        job.status = "failed"
                        return False
                    
                    print(f"⏳ Token limit exceeded during batch creation. Waiting {delay}s before retry {retries}/{self.max_retries}...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    print(f"❌ Batch creation error for {job.input_file}: {error_message}")
                    job.status = "failed"
                    return False
            except Exception as e:
                print(f"❌ Unexpected batch creation error for {job.input_file}: {str(e)}")
                job.status = "failed"
                return False
        
        return False
    
    def check_batch_status(self, job: BatchJob) -> str:
        """Check the status of a batch job."""
        if not job.batch_id:
            return job.status
        
        try:
            batch_response = self.client.batches.retrieve(job.batch_id)
            job.status = batch_response.status
            
            if batch_response.status == "completed":
                job.output_file_id = batch_response.output_file_id
            elif batch_response.status == "failed":
                job.error_file_id = batch_response.error_file_id
                
            return job.status
            
        except Exception as e:
            print(f"❌ Error checking batch status for {job.batch_id}: {str(e)}")
            return job.status
    
    def process_batch_results(self, job: BatchJob) -> bool:
        """Process and save batch results."""
        if job.status != "completed" or not job.output_file_id:
            return False
        
        try:
            # Generate output filename based on input filename
            input_filename = Path(job.input_file).stem
            output_filename = f"verification_results_{input_filename}.json"
            
            print(f"📄 Processing results for {job.input_file}...")
            
            file_response = self.client.files.content(job.output_file_id)
            raw_responses = file_response.text.strip().split('\n')
            
            verification_results = []
            error_sample_ids = []
            
            for raw_response in raw_responses:
                json_response = json.loads(raw_response)
                
                # Check for error status codes
                if json_response["response"]["status_code"] != 200:
                    error_sample_ids.append(json_response["custom_id"])
                    continue
                
                # Create verification entry
                verification_entry = {
                    "custom_id": json_response["custom_id"],
                    "verification_response": json_response["response"]["body"]["choices"][0]["message"]["content"]
                }
                verification_results.append(verification_entry)
            
            # Save results
            with open(output_filename, 'w') as f:
                json.dump(verification_results, f, indent=2)
            
            print(f"✅ Saved {len(verification_results)} verification results to {output_filename}")
            
            if error_sample_ids:
                print(f"⚠️  Error sample IDs with status_code != 200: {error_sample_ids}")
            else:
                print(f"✅ All samples processed successfully for {job.input_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing results for {job.input_file}: {str(e)}")
            return False
    
    def process_all_batches(self):
        """Main processing loop to handle all batch jobs."""
        # Discover input files
        input_files = self.discover_input_files()
        
        # Create BatchJob objects
        self.pending_jobs = [BatchJob(input_file=f) for f in input_files]
        
        print(f"🎯 Starting processing of {len(self.pending_jobs)} batch jobs")
        print(f"🔄 Max concurrent batches: {self.max_concurrent_batches}")
        
        while self.pending_jobs or self.active_jobs:
            # Start new jobs if we have capacity
            while (len(self.active_jobs) < self.max_concurrent_batches and 
                   self.pending_jobs):
                
                job = self.pending_jobs.pop(0)
                
                # Upload file
                if self.upload_file_with_retry(job):
                    # Create batch
                    if self.create_batch_with_retry(job):
                        self.active_jobs[job.batch_id] = job
                        print(f"📊 Active jobs: {len(self.active_jobs)}, Pending: {len(self.pending_jobs)}")
                    else:
                        self.failed_jobs.append(job)
                else:
                    self.failed_jobs.append(job)
            
            # Check status of active jobs
            completed_batch_ids = []
            
            for batch_id, job in self.active_jobs.items():
                status = self.check_batch_status(job)
                
                if status == "completed":
                    print(f"🎉 Batch completed: {job.input_file}")
                    if self.process_batch_results(job):
                        self.completed_jobs.append(job)
                    else:
                        self.failed_jobs.append(job)
                    completed_batch_ids.append(batch_id)
                    
                elif status == "failed":
                    print(f"💥 Batch failed: {job.input_file}")
                    self.failed_jobs.append(job)
                    completed_batch_ids.append(batch_id)
                    
                elif status in ["validating", "in_progress", "finalizing"]:
                    # Still running, check periodically
                    pass
            
            # Remove completed jobs from active list
            for batch_id in completed_batch_ids:
                del self.active_jobs[batch_id]
            
            # Print progress
            if self.active_jobs:
                print(f"⏳ {datetime.datetime.now().strftime('%H:%M:%S')} - "
                      f"Active: {len(self.active_jobs)}, "
                      f"Pending: {len(self.pending_jobs)}, "
                      f"Completed: {len(self.completed_jobs)}, "
                      f"Failed: {len(self.failed_jobs)}")
            
            # Wait before next check
            if self.active_jobs or self.pending_jobs:
                time.sleep(30)  # Check every 30 seconds
        
        # Final summary
        print("\n" + "="*50)
        print("🏁 PROCESSING COMPLETE")
        print("="*50)
        print(f"✅ Completed: {len(self.completed_jobs)}")
        print(f"❌ Failed: {len(self.failed_jobs)}")
        
        if self.failed_jobs:
            print("\n💥 Failed jobs:")
            for job in self.failed_jobs:
                print(f"  - {job.input_file} (status: {job.status})")
        
        if self.completed_jobs:
            print("\n🎉 Successfully processed files:")
            for job in self.completed_jobs:
                print(f"  - {job.input_file}")

def main():
    """Main entry point."""
    # Example usage with multiple deployments
    processor = BatchProcessor(
        verification_batches_dir="verification_batches",
        max_concurrent_batches=5,  # Adjust based on your API limits
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

def run_parallel_processors():
    """Example of running multiple processors in parallel with different deployments."""
    import threading
    
    # Define your different deployments
    deployments = [
        {
            "name": "o4-mini",
            "endpoint": "https://aisg-sj.openai.azure.com/",
            "api_key": os.getenv("AZURE_API_KEY_1")
        },
        # {
        #     "name": "o3-mini", 
        #     "endpoint": "https://decla-mbncunfi-australiaeast.cognitiveservices.azure.com/",
        #     "api_key": os.getenv("AZURE_API_KEY_2")
        # }
        # Add more deployments as needed
    ]
    
    def run_processor(deployment_config):
        """Run a single processor for a deployment."""
        try:
            print(f"🚀 Starting processor for {deployment_config['name']}")
            processor = BatchProcessor(
                verification_batches_dir=f"verification_batches_{deployment_config['name']}",
                max_concurrent_batches=3,  # Lower per processor to avoid limits
                max_retries=10,
                azure_endpoint=deployment_config['endpoint'],
                api_key=deployment_config['api_key']
            )
            processor.process_all_batches()
            print(f"✅ Processor for {deployment_config['name']} completed")
        except Exception as e:
            print(f"❌ Processor for {deployment_config['name']} failed: {str(e)}")
    
    # Start processors in parallel
    threads = []
    for deployment in deployments:
        if deployment['api_key']:  # Only start if API key is available
            thread = threading.Thread(target=run_processor, args=(deployment,))
            threads.append(thread)
            thread.start()
    
    # Wait for all to complete
    for thread in threads:
        thread.join()
    
    print("🏁 All parallel processors completed")

if __name__ == "__main__":
    main() 