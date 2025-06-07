import os
import sys
import json
import time
import logging
import threading
import queue
from datetime import datetime
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import base64
from mimetypes import guess_type

import torch
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Add the tools directory to the path
sys.path.append('/data/users/brandon/ob1-projects/InternVL/internvl_chat/tools')
from reasoning_data_pipeline.utils.accuracy_reward import check_answer, parse_answer
from reasoning_data_pipeline.utils.utils import localtime

# Azure OpenAI Configuration
endpoint = "https://dalle-declare.openai.azure.com/"
deployment = "gpt-4.1"
api_version = "2025-01-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=os.getenv("AZURE_API_KEY"),
    timeout=60.0,
)

@dataclass
class APIRequest:
    """Structured API request with metadata"""
    id: str
    messages: List[Dict]
    max_tokens: int
    temperature: float
    estimated_tokens: int
    priority: int = 0  # Higher = more important
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

class OptimalRateLimiter:
    """
    Token bucket rate limiter optimized for Azure OpenAI limits
    - 1K RPM = ~16.67 requests/second
    - 1M TPM = ~16,667 tokens/second
    """
    
    def __init__(self, rpm_limit=950, tpm_limit=950000):
        # Conservative limits with safety buffer
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        
        # Token buckets (refill every second)
        self.request_tokens = rpm_limit / 60.0  # requests per second
        self.data_tokens = tpm_limit / 60.0     # data tokens per second
        
        self.max_request_tokens = self.request_tokens * 2  # 2-second burst
        self.max_data_tokens = self.data_tokens * 2        # 2-second burst
        
        self.last_refill = time.time()
        self.lock = threading.Lock()
        
        # Stats tracking
        self.total_requests = 0
        self.total_tokens = 0
        self.throttle_count = 0
        
    def _refill_buckets(self):
        """Refill token buckets based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill
        
        if elapsed > 0:
            # Refill both buckets
            self.request_tokens = min(
                self.max_request_tokens,
                self.request_tokens + (self.rpm_limit / 60.0) * elapsed
            )
            self.data_tokens = min(
                self.max_data_tokens,
                self.data_tokens + (self.tpm_limit / 60.0) * elapsed
            )
            self.last_refill = now
    
    def can_make_request(self, estimated_tokens: int = 1000) -> bool:
        """Check if request can be made immediately"""
        with self.lock:
            self._refill_buckets()
            return (self.request_tokens >= 1.0 and 
                   self.data_tokens >= estimated_tokens)
    
    def acquire(self, estimated_tokens: int = 1000, max_wait: float = 300) -> bool:
        """
        Acquire permission to make a request (blocking)
        Returns True if acquired, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            with self.lock:
                self._refill_buckets()
                
                if self.request_tokens >= 1.0 and self.data_tokens >= estimated_tokens:
                    # Consume tokens
                    self.request_tokens -= 1.0
                    self.data_tokens -= estimated_tokens
                    self.total_requests += 1
                    self.total_tokens += estimated_tokens
                    return True
                
                # Calculate wait time until next refill
                next_refill = max(
                    (1.0 - self.request_tokens) / (self.rpm_limit / 60.0),
                    (estimated_tokens - self.data_tokens) / (self.tpm_limit / 60.0)
                )
                next_refill = min(next_refill, 1.0)  # Cap at 1 second
                
            self.throttle_count += 1
            time.sleep(next_refill)
        
        return False  # Timeout
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        with self.lock:
            self._refill_buckets()
            return {
                'request_tokens_available': self.request_tokens,
                'data_tokens_available': self.data_tokens,
                'total_requests': self.total_requests,
                'total_tokens': self.total_tokens,
                'throttle_count': self.throttle_count,
                'requests_per_second': self.rpm_limit / 60.0,
                'tokens_per_second': self.tpm_limit / 60.0,
            }

class RequestQueue:
    """Priority queue for API requests with optimal batch processing"""
    
    def __init__(self, rate_limiter: OptimalRateLimiter, max_workers: int = 50):
        self.rate_limiter = rate_limiter
        self.max_workers = max_workers
        
        # Request queue (priority queue using heapq would be more complex)
        self.pending_requests = queue.Queue()
        self.results = {}  # request_id -> result
        self.active_requests = 0
        
        # Worker pool for processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.shutdown_flag = threading.Event()
        
        # Start background processor
        self.processor_thread = threading.Thread(target=self._process_requests, daemon=True)
        self.processor_thread.start()
        
        logger.info(f"RequestQueue initialized: max_workers={max_workers}")
    
    def submit_request(self, request: APIRequest) -> None:
        """Submit a request to the queue"""
        self.pending_requests.put(request)
    
    def submit_batch(self, requests: List[APIRequest]) -> List[str]:
        """Submit a batch of requests and wait for all results"""
        # Submit all requests
        request_ids = []
        for req in requests:
            req_id = req.id
            request_ids.append(req_id)
            self.submit_request(req)
        
        # Wait for all results
        results = []
        while len(results) < len(request_ids):
            for req_id in request_ids:
                if req_id in self.results and req_id not in [r[0] for r in results]:
                    results.append((req_id, self.results[req_id]))
            
            if len(results) < len(request_ids):
                time.sleep(0.1)  # Brief wait before checking again
        
        # Sort results by original order and return content only
        results.sort(key=lambda x: request_ids.index(x[0]))
        return [result[1] for result in results]
    
    def _process_requests(self):
        """Background thread to process requests with rate limiting"""
        logger.info("Background request processor started")
        
        while not self.shutdown_flag.is_set():
            try:
                # Get next request (with timeout to allow shutdown check)
                try:
                    request = self.pending_requests.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Submit to executor when rate limit allows
                if self.rate_limiter.acquire(request.estimated_tokens):
                    future = self.executor.submit(self._execute_request, request)
                    self.active_requests += 1
                    
                    # Don't wait for result here - let it complete asynchronously
                    future.add_done_callback(lambda f: self._on_request_complete(f))
                else:
                    # Rate limit timeout - put request back
                    logger.warning(f"Rate limit timeout for request {request.id}")
                    self.pending_requests.put(request)
                
            except Exception as e:
                logger.error(f"Error in request processor: {e}")
    
    def _execute_request(self, request: APIRequest) -> Tuple[str, str]:
        """Execute a single API request"""
        try:
            response = client.chat.completions.create(
                messages=request.messages,
                max_completion_tokens=request.max_tokens,
                model=deployment,
                temperature=request.temperature,
                timeout=120.0
            )
            
            content = response.choices[0].message.content
            return request.id, content
            
        except Exception as e:
            logger.error(f"API request {request.id} failed: {e}")
            return request.id, ""
    
    def _on_request_complete(self, future):
        """Callback when request completes"""
        try:
            req_id, result = future.result()
            self.results[req_id] = result
            self.active_requests -= 1
        except Exception as e:
            logger.error(f"Request completion error: {e}")
            self.active_requests -= 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            'pending_requests': self.pending_requests.qsize(),
            'active_requests': self.active_requests,
            'completed_requests': len(self.results),
            'rate_limiter_stats': self.rate_limiter.get_stats(),
        }
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down RequestQueue...")
        self.shutdown_flag.set()
        self.processor_thread.join(timeout=10)
        self.executor.shutdown(wait=True)

# Global instances
rate_limiter = OptimalRateLimiter()
request_queue = RequestQueue(rate_limiter)

def local_image_to_data_url(image_path):
    """Convert local image to data URL"""
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'

    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    return f"data:{mime_type};base64,{base64_encoded_data}"

def estimate_tokens(prompt: str, has_image: bool = True, max_tokens: int = 4096) -> int:
    """Improved token estimation"""
    prompt_tokens = len(prompt.split()) * 1.3  # More accurate estimation
    image_tokens = 1000 if has_image else 0
    completion_tokens = max_tokens
    
    return int(prompt_tokens + image_tokens + completion_tokens)

def build_responses_optimal(
    inputs: List[Tuple[str, str]], 
    num_return_sequences: int = 1,
    prefixes: Optional[List[str]] = None,
    max_new_tokens: int = 4096,
    temperature: float = 1.0
) -> List[str]:
    """
    Optimally build responses using the request queue system
    """
    total_requests = len(inputs) * num_return_sequences
    logger.info(f"Building {total_requests} responses with optimal rate limiting")
    
    # Prepare all requests
    requests = []
    request_id_counter = 0
    
    for seq_idx in range(num_return_sequences):
        for input_idx, (prompt, image_path) in enumerate(inputs):
            # Prepare message content
            try:
                data_url = local_image_to_data_url(image_path)
            except Exception as e:
                logger.error(f"Failed to process image {image_path}: {e}")
                continue
            
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant that excels at visual reasoning and pattern recognition."},
                {"role": "user", "content": content}
            ]
            
            # Add prefix if provided
            if prefixes and input_idx < len(prefixes) and prefixes[input_idx]:
                messages.append({"role": "assistant", "content": prefixes[input_idx]})
            
            # Create request
            estimated_tokens = estimate_tokens(prompt, True, max_new_tokens)
            request = APIRequest(
                id=f"req_{request_id_counter}",
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                estimated_tokens=estimated_tokens,
                priority=0  # Could prioritize shorter requests
            )
            
            requests.append(request)
            request_id_counter += 1
    
    # Process requests in optimal batches
    batch_size = min(100, len(requests))  # Process in batches of 100
    responses = []
    
    for i in range(0, len(requests), batch_size):
        batch = requests[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}: {len(batch)} requests")
        
        batch_start = time.time()
        batch_responses = request_queue.submit_batch(batch)
        batch_duration = time.time() - batch_start
        
        responses.extend(batch_responses)
        
        logger.info(f"Batch completed in {batch_duration:.2f}s ({len(batch)/batch_duration:.2f} req/s)")
        
        # Log rate limiter stats periodically
        if (i // batch_size + 1) % 5 == 0:
            stats = request_queue.get_stats()
            logger.info(f"Queue stats: {stats}")
    
    # Reconstruct in correct order
    response_list = []
    for input_idx in range(len(inputs)):
        for seq_idx in range(num_return_sequences):
            idx = seq_idx * len(inputs) + input_idx
            if idx < len(responses):
                response_list.append(responses[idx])
            else:
                response_list.append("")
    
    return response_list

class RAVENDataset(torch.utils.data.Dataset):
    def __init__(self, data, sample_max_num=None, sample_start_idx=0):
        with open(data, 'r', encoding='utf-8') as file:
            self.data = file.readlines()

        if sample_max_num is not None and len(self.data) > sample_max_num:
            print(f'Truncate data lines. {len(self.data)} => {sample_max_num}')
            step = max(len(self.data) // sample_max_num, 1)
            self.data = self.data[sample_start_idx::step][:sample_max_num]
            print(f'Number of data lines after truncation: {len(self.data)=}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = json.loads(self.data[idx])
        
        image_path = item['combined_image_path']
        correct_answer = item['correct_answer']
        
        rollout_user_prompt = r"""You are an abstract reasoning puzzle expert. The puzzle you will receive is presented in a standard Raven's Progressive Matrices format: a 3×3 matrix of related images, with the bottom-right cell (the ninth tile) missing. There are eight possible answer choices provided separately, and your task is to decide which of those eight images correctly completes the 3×3 matrix pattern.

I will provide you with an image containing:
- Problem Matrix: An accompanying image that shows the eight tiles and highlights where the ninth is missing.
- Answer Set: The eight candidate images from which you must choose the best fit for the missing tile.

Your task is to:
- Review the problem matrix and the accompanying image in sequence, describing step-by-step what you see in the image in <perception> tags.
- Reason step-by-step about the logical pattern or rule connecting the tiles in <reasoning> tags.
- Deduce the correct tile from the eight provided options in <correct_answer> tags.

It is crucial that your solution contains these sections in the exact format described below:

```
[Perception]
<step_1>
...(Step 1 of step-by-step perception)...
</step_1>
<step_2>
...(Step 2 of step-by-step perception)...
</step_2>
...
<step_n>
...(Step n of step-by-step perception)...
</step_n>

[Reasoning]
<step_1>
...(Step 1 of step-by-step reasoning)...
</step_1>
<step_2>
...(Step 2 of step-by-step reasoning)...
</step_2>
...
<step_m>
...(Step m of step-by-step reasoning)...
</step_m>

<correct_answer>
...(Clearly state which of the 8 candidate images is the best candidate image as the missing tile to complete the matrix. If the candidates are numbered, lettered, or can be uniquely described, use that identifier.)...
</correct_answer>
```
"""

        return {
            'rollout_user_prompt': rollout_user_prompt,
            'image': image_path,
            'image_path': image_path,
            'item': item.copy(),
            'correct_answer': correct_answer,
        }

def parse_response_to_steps(text, max_perception_steps=12, max_reasoning_steps=12):
    """Parse response into perception and reasoning steps"""
    import re
    
    result = {
        'perception_steps': [],
        'reasoning_steps': [],
        'llm_answer': None
    }
    
    # Extract perception steps
    perception_pattern = r'\[Perception\](.*?)(?=\[Reasoning\]|\Z)'
    perception_match = re.search(perception_pattern, text, re.DOTALL)
    
    if perception_match:
        perception_text = perception_match.group(1).strip()
        step_pattern = r'<step_(\d+)>(.*?)</step_\1>'
        perception_steps = re.findall(step_pattern, perception_text, re.DOTALL)
        perception_steps.sort(key=lambda x: int(x[0]))
        result['perception_steps'] = [step[1].strip() for step in perception_steps]
    
    # Extract reasoning steps
    reasoning_pattern = r'\[Reasoning\](.*?)(?=<correct_answer>|\Z)'
    reasoning_match = re.search(reasoning_pattern, text, re.DOTALL)
    
    if reasoning_match:
        reasoning_text = reasoning_match.group(1).strip()
        step_pattern = r'<step_(\d+)>(.*?)</step_\1>'
        reasoning_steps = re.findall(step_pattern, reasoning_text, re.DOTALL)
        reasoning_steps.sort(key=lambda x: int(x[0]))
        result['reasoning_steps'] = [step[1].strip() for step in reasoning_steps]
    
    # Extract correct answer
    answer_pattern = r'<correct_answer>(.*?)</correct_answer>'
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    
    if answer_match:
        result['llm_answer'] = answer_match.group(1).strip()
    
    return result

def maximize_throughput_simple():
    """Fire 950 requests every 60 seconds - period."""
    
    while tasks_remaining:
        batch = get_next_950_tasks()
        
        # Simple time-based firing
        time.sleep_until(next_minute_boundary)
        
        # Fire all 950 requests in parallel
        with ThreadPoolExecutor(max_workers=100):
            results = [make_request(task) for task in batch]
        
        process_results(results)

def main():
    """Main execution function"""
    # Configuration with optimal parameters
    args = {
        'prompt_path': '/data/users/brandon/ob1-projects/InternVL/internvl_chat/rollout_generation/preprocessed_prompts/preprocessing_scripts/RAVEN/raven_processed_jsonl/center_single_train.jsonl',
        'out_dir': 'raven_rollouts_output_optimal',
        'batch_size': 16,  # Optimal: 16×2 = 32 initial requests per batch
        'num_return_sequences': 2,
        'sample_start_idx': 0,
        'sample_max_num': 100,  # Small test set
        'num_mc_sequences': 4,  # Reduced for faster processing
        'max_new_tokens': 4096,
        'temperature': 1.0,
    }
    
    # Setup logging
    os.makedirs(args['out_dir'], exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args['out_dir'], f'optimal_rollout_{timestamp}.log')
    
    global logger
    logger = logging.getLogger('optimal_rollout')
    logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Starting optimal RAVEN rollout generation")
    logger.info(f"Configuration: {args}")
    
    # Load dataset
    dataset = RAVENDataset(
        data=args['prompt_path'],
        sample_max_num=args['sample_max_num'],
        sample_start_idx=args['sample_start_idx'],
    )
    
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    
    # Process in batches
    batch_size = args['batch_size']
    outputs = []
    
    total_start = time.time()
    
    for batch_start in range(0, len(dataset), batch_size):
        batch_end = min(batch_start + batch_size, len(dataset))
        batch_samples = [dataset[i] for i in range(batch_start, batch_end)]
        
        logger.info(f"Processing batch {batch_start//batch_size + 1}: samples {batch_start+1}-{batch_end}")
        
        # Prepare batch inputs
        batch_inputs = [(sample['rollout_user_prompt'], sample['image']) for sample in batch_samples]
        
        # Generate initial rollouts
        batch_start_time = time.time()
        responses = build_responses_optimal(
            batch_inputs,
            num_return_sequences=args['num_return_sequences'],
            max_new_tokens=args['max_new_tokens'],
            temperature=args['temperature']
        )
        batch_duration = time.time() - batch_start_time
        
        logger.info(f"Batch rollouts completed in {batch_duration:.2f}s")
        
        # Process responses
        for i, response in enumerate(responses):
            sample_idx = batch_start + (i // args['num_return_sequences'])
            sample = batch_samples[sample_idx]
            
            output = sample['item'].copy()
            output['response'] = response
            output['question'] = sample['rollout_user_prompt']
            
            # Parse steps (simplified - no MC evaluation for this demo)
            try:
                steps = parse_response_to_steps(response)
                output['steps'] = {
                    'perception_steps': steps['perception_steps'],
                    'reasoning_steps': steps['reasoning_steps'],
                    'llm_answer': steps['llm_answer']
                }
            except Exception as e:
                logger.error(f"Failed to parse response: {e}")
                output['steps'] = {'error': str(e)}
            
            outputs.append(output)
        
        # Save incrementally
        output_file = os.path.join(args['out_dir'], f'optimal_rollouts_{args["sample_start_idx"]}_{args["sample_max_num"]}.jsonl')
        file_mode = 'w' if batch_start == 0 else 'a'
        with open(output_file, file_mode, encoding='utf-8') as f:
            for output in outputs[-len(responses):]:  # Only write new outputs
                f.write(json.dumps(output, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved batch outputs (total: {len(outputs)})")
    
    total_duration = time.time() - total_start
    logger.info(f"Processing completed in {total_duration:.2f}s")
    logger.info(f"Average time per sample: {total_duration/len(outputs):.2f}s")
    
    # Final stats
    final_stats = request_queue.get_stats()
    logger.info(f"Final queue stats: {final_stats}")
    
    # Cleanup
    request_queue.shutdown()

if __name__ == "__main__":
    main() 