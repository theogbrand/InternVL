import os
import sys
import json
import math
import base64
import time
import logging
import asyncio
import signal
from datetime import datetime
from mimetypes import guess_type
from collections import defaultdict
import torch
from PIL import Image
from openai import AzureOpenAI
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log
)
from concurrent.futures import ThreadPoolExecutor
import threading

# Add the tools directory to the path
sys.path.append('/data/users/brandon/ob1-projects/InternVL/internvl_chat/tools')

from reasoning_data_pipeline.utils.accuracy_reward import (check_answer, parse_answer)

from reasoning_data_pipeline.utils.utils import localtime

# Azure OpenAI Configuration
endpoint = "https://dalle-declare.openai.azure.com/"
deployment = "gpt-4.1"
api_version = "2025-01-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=os.getenv("AZURE_API_KEY"),
    timeout=60.0,  # 60 second timeout
)

# Rate limiting configuration with dynamic optimization
class RateLimiter:
    def __init__(self, max_requests_per_minute=900, max_tokens_per_minute=950000):
        self.max_rpm = max_requests_per_minute  # Conservative buffer
        self.max_tpm = max_tokens_per_minute   # Conservative buffer
        self.request_times = []
        self.token_usage = []
        self.lock = threading.Lock()
        
        # Performance tracking
        self.success_count = 0
        self.failure_count = 0
        self.last_optimization = time.time()
    
    def can_make_request(self, estimated_tokens=1000):
        with self.lock:
            now = time.time()
            minute_ago = now - 60
            
            # Clean old entries
            self.request_times = [t for t in self.request_times if t > minute_ago]
            self.token_usage = [(t, tokens) for t, tokens in self.token_usage if t > minute_ago]
            
            # Check limits
            current_rpm = len(self.request_times)
            current_tpm = sum(tokens for _, tokens in self.token_usage)
            
            return (current_rpm < self.max_rpm and 
                   current_tpm + estimated_tokens < self.max_tpm)
    
    def record_request(self, tokens_used=1000, success=True):
        with self.lock:
            now = time.time()
            self.request_times.append(now)
            self.token_usage.append((now, tokens_used))
            
            if success:
                self.success_count += 1
            else:
                self.failure_count += 1
            
            # Optimize limits every 5 minutes based on performance
            if now - self.last_optimization > 300:  # 5 minutes
                self._optimize_limits()
                self.last_optimization = now
    
    def _optimize_limits(self):
        """Dynamically adjust limits based on success rate"""
        total_requests = self.success_count + self.failure_count
        if total_requests > 0:
            success_rate = self.success_count / total_requests
            
            if success_rate > 0.98:  # Very high success, increase limits slightly
                self.max_rpm = min(950, int(self.max_rpm * 1.05))
                self.max_tpm = min(980000, int(self.max_tpm * 1.02))
                if 'logger' in globals():
                    logger.info(f"Optimized rate limits: RPM={self.max_rpm}, TPM={self.max_tpm}")
                else:
                    print(f"Optimized rate limits: RPM={self.max_rpm}, TPM={self.max_tpm}")
            elif success_rate < 0.90:  # Low success, decrease limits
                self.max_rpm = max(500, int(self.max_rpm * 0.9))
                self.max_tpm = max(800000, int(self.max_tpm * 0.95))
                if 'logger' in globals():
                    logger.warning(f"Reduced rate limits due to failures: RPM={self.max_rpm}, TPM={self.max_tpm}")
                else:
                    print(f"WARNING: Reduced rate limits due to failures: RPM={self.max_rpm}, TPM={self.max_tpm}")
    
    def wait_if_needed(self, estimated_tokens=1000):
        wait_time = 0
        max_wait_time = 300  # Maximum wait time of 5 minutes to prevent infinite hanging
        
        while not self.can_make_request(estimated_tokens):
            if wait_time >= max_wait_time:
                raise TimeoutError(f"Rate limiting wait exceeded {max_wait_time}s - possible deadlock")
            
            time.sleep(0.5)  # More responsive checking
            wait_time += 0.5
            if wait_time > 30 and wait_time % 30 == 0:  # Log every 30 seconds after initial 30s
                if 'logger' in globals():
                    logger.warning(f"Rate limiting: waited {wait_time}s for tokens={estimated_tokens}")
                else:
                    print(f"WARNING: Rate limiting: waited {wait_time}s for tokens={estimated_tokens}")
                    sys.stdout.flush()  # Ensure output is written in screen sessions
    
    def get_stats(self):
        """Get current rate limiting statistics"""
        with self.lock:
            now = time.time()
            minute_ago = now - 60
            current_rpm = len([t for t in self.request_times if t > minute_ago])
            current_tpm = sum(tokens for t, tokens in self.token_usage if t > minute_ago)
            total_requests = self.success_count + self.failure_count
            success_rate = self.success_count / total_requests if total_requests > 0 else 0
            
            return {
                'current_rpm': current_rpm,
                'max_rpm': self.max_rpm,
                'current_tpm': current_tpm,
                'max_tpm': self.max_tpm,
                'success_rate': success_rate,
                'total_requests': total_requests
            }

# Global rate limiter
rate_limiter = RateLimiter()

# Global shutdown flag for graceful termination
shutdown_flag = threading.Event()

def signal_handler(signum, frame):
    """Handle termination signals gracefully"""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_flag.set()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Function to encode a local image into data URL 
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

class RAVENDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        sample_max_num=None,
        sample_start_idx=0,
    ):
        with open(data) as file:
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
        
        # RAVEN dataset structure: id, combined_image_path, correct_answer, subset_split
        image_path = item['combined_image_path']
        correct_answer = item['correct_answer']
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
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
            'image': image,
            'image_path': image_path,
            'item': item.copy(),
            'correct_answer': correct_answer,
        }


def parse_response_to_perception_and_reasoning_steps_and_correct_answer(text, max_perception_steps=None, max_reasoning_steps=None):
    """
    Parse text that contains perception steps, reasoning steps, and a correct answer.
    
    Args:
        text (str): The text to parse
        
    Returns:
        dict: Dictionary with 'perception_steps', 'reasoning_steps', and 'llm_answer'
        
    Raises:
        ValueError: If the text doesn't contain all required sections
    """
    import re
    
    # Initialize the result dictionary
    result = {
        'perception_steps': [],
        'reasoning_steps': [],
        'llm_answer': None
    }
    
    # Extract perception steps
    perception_pattern = r'\[Perception\](.*?)(?=\[Reasoning\]|\Z)'
    perception_match = re.search(perception_pattern, text, re.DOTALL)
    
    if not perception_match:
        raise ValueError("Could not find Perception section")
    
    perception_text = perception_match.group(1).strip()
    step_pattern = r'<step_(\d+)>(.*?)</step_\1>'
    perception_steps = re.findall(step_pattern, perception_text, re.DOTALL)
    
    if not perception_steps:
        raise ValueError("Could not find any perception steps")
    
    # Sort by step number and extract content
    perception_steps.sort(key=lambda x: int(x[0]))
    result['perception_steps'] = [step[1].strip() for step in perception_steps]
    
    # Extract reasoning steps
    reasoning_pattern = r'\[Reasoning\](.*?)(?=<correct_answer>|\Z)'
    reasoning_match = re.search(reasoning_pattern, text, re.DOTALL)
    
    if not reasoning_match:
        raise ValueError("Could not find Reasoning section")
    
    reasoning_text = reasoning_match.group(1).strip()
    reasoning_steps = re.findall(step_pattern, reasoning_text, re.DOTALL)
    
    if not reasoning_steps:
        raise ValueError("Could not find any reasoning steps")
    
    # Sort by step number and extract content
    reasoning_steps.sort(key=lambda x: int(x[0]))
    result['reasoning_steps'] = [step[1].strip() for step in reasoning_steps]
    
    # Extract correct answer
    answer_pattern = r'<correct_answer>(.*?)</correct_answer>'
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    
    if not answer_match:
        raise ValueError("Could not find correct answer")
    
    result['llm_answer'] = answer_match.group(1).strip()
    
    # Final validation to ensure we have all components
    if not result['perception_steps'] or not result['reasoning_steps'] or not result['llm_answer']:
        raise ValueError("Missing one or more required components")
    
    return result

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((Exception,))
)
def make_azure_request(messages, max_tokens, temperature, estimated_tokens=1000):
    """Make a rate-limited Azure OpenAI request with retry logic"""
    # Wait for rate limit if needed
    try:
        rate_limiter.wait_if_needed(estimated_tokens)
    except TimeoutError as e:
        logger.error(f"Rate limiting timeout: {e}")
        raise
    
    try:
        start_time = time.time()
        response = client.chat.completions.create(
            messages=messages,
            max_completion_tokens=max_tokens,
            model=deployment,
            temperature=temperature,
            timeout=120.0  # Increased timeout for complex requests
        )
        
        # Record successful request for rate limiting
        actual_tokens = response.usage.total_tokens if hasattr(response, 'usage') else estimated_tokens
        rate_limiter.record_request(actual_tokens, success=True)
        
        duration = time.time() - start_time
        # Use print if logger not available yet, otherwise use logger
        if 'logger' in globals():
            logger.debug(f"API call completed in {duration:.2f}s, tokens: {actual_tokens}")
        
        return response.choices[0].message.content
        
    except Exception as e:
        # Record failed request
        rate_limiter.record_request(estimated_tokens, success=False)
        # Use print if logger not available yet, otherwise use logger
        if 'logger' in globals():
            logger.error(f"API request failed: {e}")
        else:
            print(f"API request failed: {e}")
            sys.stdout.flush()
        raise

def build_responses_azure_parallel(inputs, num_return_sequences=1, prefixes=None, max_new_tokens=4096, temperature=1.0, max_workers=20):
    """
    Build responses using Azure OpenAI GPT-4.1 with parallel processing
    """
    total_requests = len(inputs) * num_return_sequences
    if 'logger' in globals():
        logger.info(f"Starting parallel processing of {total_requests} requests with {max_workers} workers")
    else:
        print(f"Starting parallel processing of {total_requests} requests with {max_workers} workers")
    
    def process_single_request(args_tuple):
        input_idx, seq_idx, prompt, image, prefix = args_tuple
        
        # Check for shutdown signal
        if shutdown_flag.is_set():
            return (input_idx, seq_idx, "")
        
        try:
            # Convert image to data URL
            temp_path = None
            try:
                if isinstance(image, str):
                    data_url = local_image_to_data_url(image)
                else:
                    temp_path = f"/tmp/temp_image_{input_idx}_{seq_idx}_{threading.current_thread().ident}.png"
                    image.save(temp_path)
                    data_url = local_image_to_data_url(temp_path)
            except Exception as e:
                if 'logger' in globals():
                    logger.error(f"Failed to process image for input_idx={input_idx}, seq_idx={seq_idx}: {e}")
                else:
                    print(f"Failed to process image for input_idx={input_idx}, seq_idx={seq_idx}: {e}")
                raise
            finally:
                # Clean up temp file if it was created
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception as e:
                        if 'logger' in globals():
                            logger.warning(f"Failed to clean up temp file {temp_path}: {e}")
                        else:
                            print(f"Warning: Failed to clean up temp file {temp_path}: {e}")
            
            # Prepare messages
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant that excels at visual reasoning and pattern recognition."},
                {"role": "user", "content": content}
            ]
            
            # Add prefix if provided
            if prefix:
                messages.append({"role": "assistant", "content": prefix})
            
            # Estimate tokens (prompt + image + completion) with safer bounds
            prompt_tokens = min(len(prompt) // 4, 8000)  # Cap prompt estimation
            image_tokens = 1000  # Standard image token cost
            estimated_tokens = prompt_tokens + image_tokens + max_new_tokens
            
            response_text = make_azure_request(messages, max_new_tokens, temperature, estimated_tokens)
            
            return (input_idx, seq_idx, response_text)
            
        except Exception as e:
            if 'logger' in globals():
                logger.error(f"Error in request input_idx={input_idx}, seq_idx={seq_idx}: {e}")
            else:
                print(f"Error in request input_idx={input_idx}, seq_idx={seq_idx}: {e}")
            return (input_idx, seq_idx, "")
    
    # Prepare all request arguments
    request_args = []
    for seq_idx in range(num_return_sequences):
        for input_idx, (prompt, image) in enumerate(inputs):
            prefix = prefixes[input_idx] if prefixes else None
            request_args.append((input_idx, seq_idx, prompt, image, prefix))
    
    # Process requests in parallel with timeout handling
    results = {}
    
    # Start heartbeat for long-running operations
    last_heartbeat = time.time()
    heartbeat_interval = 300  # 5 minutes
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all requests
        future_to_args = {
            executor.submit(process_single_request, args): args 
            for args in request_args
        }
        
        # Collect results with progress bar and timeout handling
        if 'tqdm' in globals() and not shutdown_flag.is_set():
            pbar = tqdm(total=len(request_args), desc="API Requests", unit="req", disable=not sys.stdout.isatty())
        else:
            pbar = None
        
        completed_count = 0
        for future in future_to_args:
            # Check for shutdown signal
            if shutdown_flag.is_set():
                logger.info("Shutdown signal received, cancelling remaining requests...")
                break
                
            # Heartbeat mechanism to prevent silent hanging
            current_time = time.time()
            if current_time - last_heartbeat > heartbeat_interval:
                if 'logger' in globals():
                    logger.info(f"Heartbeat: {completed_count}/{len(request_args)} requests completed")
                    sys.stdout.flush()
                last_heartbeat = current_time
            
            try:
                # Add timeout to prevent hanging indefinitely
                input_idx, seq_idx, response = future.result(timeout=600)  # 10 minute timeout per request
                results[(input_idx, seq_idx)] = response
                completed_count += 1
                
                if pbar:
                    pbar.update(1)
                elif completed_count % 10 == 0:  # Log progress every 10 completions
                    if 'logger' in globals():
                        logger.info(f"API Progress: {completed_count}/{len(request_args)} requests completed")
                        sys.stdout.flush()
                        
            except Exception as e:
                args = future_to_args[future]
                if 'logger' in globals():
                    logger.error(f"Future failed for args {args}: {e}")
                else:
                    print(f"Future failed for args {args}: {e}")
                results[(args[0], args[1])] = ""
                completed_count += 1
                
                if pbar:
                    pbar.update(1)
        
        if pbar:
            pbar.close()
    
    # Reconstruct response list in correct order
    response_list = []
    for input_idx in range(len(inputs)):
        for seq_idx in range(num_return_sequences):
            response = results.get((input_idx, seq_idx), "")
            response_list.append(response)
    
    # Validation: ensure we have the expected number of responses
    expected_count = len(inputs) * num_return_sequences
    actual_count = len(response_list)
    successful_count = len([r for r in response_list if r.strip()])
    
    if 'logger' in globals():
        logger.info(f"Response validation: Expected={expected_count}, Actual={actual_count}, Successful={successful_count}")
    else:
        print(f"Response validation: Expected={expected_count}, Actual={actual_count}, Successful={successful_count}")
    
    if actual_count != expected_count:
        error_msg = f"Mismatch in response count! Expected {expected_count}, got {actual_count}"
        if 'logger' in globals():
            logger.error(error_msg)
        else:
            print(f"ERROR: {error_msg}")
        raise ValueError(f"Response count mismatch: expected {expected_count}, got {actual_count}")
    
    if successful_count < expected_count * 0.95:  # Allow 5% failure rate
        warning_msg = f"High failure rate: {expected_count - successful_count} failed out of {expected_count}"
        if 'logger' in globals():
            logger.warning(warning_msg)
        else:
            print(f"WARNING: {warning_msg}")
    
    return response_list

# Updated function call
def build_responses_azure(inputs, num_return_sequences=1, prefixes=None, max_new_tokens=4096, temperature=1.0):
    """Wrapper for backward compatibility"""
    # Get max_workers from global args if available
    max_workers = args.get('max_workers', 20) if 'args' in globals() else 20
    return build_responses_azure_parallel(
        inputs, num_return_sequences, prefixes, max_new_tokens, temperature, max_workers
    )

def build_mc_scores(inputs, response_list, items, num_return_sequences, args):
    """
    Build Monte Carlo scores for each step in the responses
    """
    assert len(response_list) == len(inputs) * num_return_sequences

    steps_list = []
    for response in response_list:
        try:
            steps = parse_response_to_perception_and_reasoning_steps_and_correct_answer(response, max_perception_steps=args.get('max_perception_steps', 12), max_reasoning_steps=args.get('max_reasoning_steps', 12))
            steps_list.append(steps)
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            # Add dummy structure to maintain indexing
            steps_list.append({'perception_steps': ['Error parsing'], 'reasoning_steps': ['Error parsing'], 'llm_answer': 'Error'})
    
    # Convert structured steps to flat lists for processing
    flat_steps_list = []
    for steps_dict in steps_list:
        # Combine perception and reasoning steps into a single flat list
        flat_steps = steps_dict['perception_steps'] + steps_dict['reasoning_steps']
        flat_steps_list.append(flat_steps)
    
    steps_flag = [False for _ in range(len(response_list))]
    steps_outputs = [[] for _ in range(len(response_list))]

    step_cnt = 0
    while True:
        logger.info(f"=== STEP_CNT = {step_cnt} ===")
        curr_inputs_idx = []
        curr_inputs = []
        curr_prefixes = []
        curr_answer_gt = []
        
        for idx, (flat_steps, flag) in enumerate(zip(flat_steps_list, steps_flag)):
            if step_cnt >= len(flat_steps):
                continue

            if flag:
                steps_outputs[idx].append({
                    'step': flat_steps[step_cnt],
                    'score': 0.0,
                    'num_mc_correct': 0,
                    'num_mc_total': 0,
                })
                continue
            
            # With 2 inputs, num_return_sequences = 2: [input_A, input_B] 
            # response_list: [resp_A1, resp_A2, resp_B1, resp_B2] (length 4)

            # idx=0 → inputs[0 // 2] = inputs[0] = input_A ✓
            # idx=1 → inputs[1 // 2] = inputs[0] = input_A ✓  
            # idx=2 → inputs[2 // 2] = inputs[1] = input_B ✓
            # idx=3 → inputs[3 // 2] = inputs[1] = input_B ✓
            input = inputs[idx // num_return_sequences]
            item = items[idx // num_return_sequences]

            # only add to curr_inputs if this generated response needs MC evaluation
            curr_inputs_idx.append(idx)
            curr_inputs.append(input)
            
            # Build prefix: perception + reasoning up to current step
            prefix_steps = flat_steps[:step_cnt+1]
            
            # Reconstruct the proper format for the prefix
            perception_count = len(steps_list[idx]['perception_steps'])
            if step_cnt < perception_count:
                # We're still in perception steps
                perception_prefix = prefix_steps
                reasoning_prefix = []
            else:
                # We're in reasoning steps
                perception_prefix = steps_list[idx]['perception_steps']
                reasoning_prefix = prefix_steps[perception_count:]
            
            # Format the prefix properly
            formatted_prefix = "" # add the Perception and/or reasoning step to analyse here
            if perception_prefix:
                formatted_prefix += "[Perception]\n"
                for i, step in enumerate(perception_prefix):
                    formatted_prefix += f"<step_{i+1}>\n{step}\n</step_{i+1}>\n"
                formatted_prefix += "\n"
            
            if reasoning_prefix:
                formatted_prefix += "[Reasoning]\n"
                for i, step in enumerate(reasoning_prefix):
                    formatted_prefix += f"<step_{i+1}>\n{step}\n</step_{i+1}>\n"
            
            curr_answer_gt.append(item['correct_answer'])
            
            logger.debug("--------------------------------")
            logger.debug(f"added current formatted_prefix for this step_cnt {formatted_prefix}")
            logger.debug("--------------------------------")
            # why we append the formatted_prefix to another array (curr_prefixes) is so we separate this "processed input" from the input object (curr_inputs) which are the original inputs. 
            # we use curr_prefixes to add to the assisstant message to "prefill" the assistant's response
            curr_prefixes.append(formatted_prefix.strip())

        # used for managing batch processing of inputs where sometimes there will be no curr_inputs to process, and we might need to add a zero score step.
        if len(curr_inputs) <= 0:
            for idx, flat_steps in enumerate(flat_steps_list):
                for step_idx in range(len(flat_steps) - step_cnt - 1):
                    steps_outputs[idx].append({
                        'step': flat_steps[step_cnt + step_idx + 1],
                        'score': 0.0,
                        'num_mc_correct': 0,
                        'num_mc_total': 0,
                    })
            break

        # Here is where based on the current steps (prefixes), we rollout to get mc soft estimation. 
        logger.info(f"Starting Monte Carlo rollouts for step_cnt {step_cnt} with {len(curr_inputs)} inputs")
        mc_start_time = time.time()
        
        mc_response_list = build_responses_azure(
            curr_inputs, 
            args.get('num_mc_sequences', 16), 
            curr_prefixes,
            max_new_tokens=args.get('max_new_tokens', 4096),
            temperature=args.get('temperature', 1.0)
        )
        
        mc_duration = time.time() - mc_start_time
        logger.info(f"Monte Carlo rollouts completed in {mc_duration:.2f} seconds for step_cnt {step_cnt}")

        # Validation: Check expected vs actual MC responses
        expected_mc_count = len(curr_inputs) * args.get('num_mc_sequences', 16)
        actual_mc_count = len(mc_response_list)
        successful_mc_count = len([r for r in mc_response_list if r.strip()])
        
        logger.info(f"MC Response validation: Expected={expected_mc_count}, Actual={actual_mc_count}, Successful={successful_mc_count}")
        
        if actual_mc_count != expected_mc_count:
            logger.error(f"MC Response count mismatch! Expected {expected_mc_count}, got {actual_mc_count}")
            raise ValueError(f"MC Response count mismatch: expected {expected_mc_count}, got {actual_mc_count}")
        
        success_rate = successful_mc_count / expected_mc_count
        if success_rate < args.get('validation_threshold', 0.95):
            logger.warning(f"Low MC success rate: {success_rate:.3f} ({successful_mc_count}/{expected_mc_count})")
        
        logger.debug(f"mc_response_list in build_mc_scores for step_cnt {step_cnt} has {len(mc_response_list)} responses")
        logger.info(f"MC throughput: {len(mc_response_list) / mc_duration:.2f} responses/second")

        # Here is where we get the correctness of the rollouts to label the corresponding rollout as correct or incorrect. 
        correctness_list = []
        for mc_idx, mc_response in enumerate(mc_response_list):
            try:
                # For RAVEN, we need to extract the final answer (number 1-8)
                logger.debug(f"checking answer correctness for mc_response for this step_cnt {step_cnt} is {mc_response}")
                correctness = check_answer(
                    answer_pred=parse_answer(mc_response, prompt_version=args.get('prompt_version', 'raven_v1'))[-1], # parse based on answer format
                    answer_gt=str(curr_answer_gt[mc_idx // args.get('num_mc_sequences', 16)]),
                    mode='raven_score'  # Use RAVEN verification mode
                )
            except Exception as e:
                logger.error(f'Fail to check correctness for response: {mc_response[:100]}... Error: {e}')
                correctness = 0
            correctness_list.append(correctness)
        logger.info(f"correctness_list for this step_cnt {step_cnt} is {correctness_list}")

        assert len(mc_response_list) == len(correctness_list)
        assert len(mc_response_list) == len(curr_inputs) * args.get('num_mc_sequences', 16)

        for idx_idx, idx in enumerate(curr_inputs_idx):
            curr_correctness_list = correctness_list[idx_idx*args.get('num_mc_sequences', 16):(idx_idx+1)*args.get('num_mc_sequences', 16)]
            score = sum(curr_correctness_list) / len(curr_correctness_list)
            steps_outputs[idx].append({
                'step': flat_steps_list[idx][step_cnt],
                'score': score,
                'num_mc_correct': sum(curr_correctness_list),
                'num_mc_total': len(curr_correctness_list),
            })

            if score == 0 and args.get('early_stop', True):
                steps_flag[idx] = True

        step_cnt += 1
    return steps_outputs

def build_process_supervision(inputs, items, num_return_sequences, args):
    """
    Build process supervision data with step-by-step scoring
    """
    logger.info(f"Starting initial rollout generation for {len(inputs)} inputs with {num_return_sequences} sequences each")
    initial_start_time = time.time()
    
    response_list = build_responses_azure(
        inputs, # rollout_user_prompt, image
        num_return_sequences,
        max_new_tokens=args.get('max_new_tokens', 4096),
        temperature=args.get('temperature', 1.0)
    )
    
    initial_duration = time.time() - initial_start_time
    logger.info(f"Initial rollout generation completed in {initial_duration:.2f} seconds")

    logger.debug(f"responses produced by build_process_supervision {response_list}") # num_return_sequences = n, so n rollouts for each input image. 
    steps_with_score = build_mc_scores(inputs, response_list, items, num_return_sequences, args)
    # return

    outputs = []

    for idx, (response, each_steps_with_score) in enumerate(zip(response_list, steps_with_score)):
        input = inputs[idx // num_return_sequences]
        item = items[idx // num_return_sequences]

        output = item.copy()
        output['response'] = response
        output['steps_with_score'] = each_steps_with_score
        output['question'] = input[0]  # Store the formatted question
        outputs.append(output)

    return outputs


def print_process_supervision(output):
    """
    Print process supervision output for debugging
    """
    steps_with_score = output['steps_with_score']
    logger.info('[Response] Start')
    for step_idx, step in enumerate(steps_with_score):
        logger.info(
            f'[Steps-{step_idx}] Start\n'
            f"{step['step']}\n\n"
            f"Score: {step['score']}\n"
            f"MC Correct: {step['num_mc_correct']}\n"
            f"MC Total: {step['num_mc_total']}\n"
            f'[Steps-{step_idx}] End\n'
        )
    logger.info('[Response] End')


# Configuration parameters optimized for maximum throughput
args = {
    'prompt_path': '/data/users/brandon/ob1-projects/InternVL/internvl_chat/rollout_generation/preprocessed_prompts/preprocessing_scripts/RAVEN/raven_processed_jsonl/center_single_train.jsonl',
    'out_dir': 'raven_rollouts_output',
    'batch_size': 50,  # Optimized for parallel processing within rate limits
    'num_return_sequences': 2,
    'sample_start_idx': 0,
    'sample_max_num': 1200,  # Limit for testing
    'prompt_version': 'raven_v1',
    'num_mc_sequences': 8,  # Increased for better MC estimation
    'max_perception_steps': 12,
    'max_reasoning_steps': 12,
    'early_stop': True, # when a step results in an incorrect answer we immediately stop rollouts from THAT step.
    'max_new_tokens': 4096,
    'temperature': 1.0,
    'max_workers': 20,  # Parallel processing threads
    'validation_threshold': 0.95,  # Minimum success rate required
}

# Create output directory
os.makedirs(args['out_dir'], exist_ok=True)

# Setup logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"raven_rollout_{timestamp}_samples_{args['sample_start_idx']}_{args['sample_max_num']}.log"
log_filepath = os.path.join(args['out_dir'], log_filename)

# Create custom logger
logger = logging.getLogger('raven_rollout')
logger.setLevel(logging.DEBUG)  # Allow all levels to reach handlers

# Create formatters
detailed_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
simple_formatter = logging.Formatter('%(message)s')

# File handler - detailed logging (includes DEBUG)
file_handler = logging.FileHandler(log_filepath)
file_handler.setLevel(logging.DEBUG)  # Save all debug info to file
file_handler.setFormatter(detailed_formatter)

# Console handler - replace TqdmLoggingHandler with StreamHandler for screen compatibility
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)  # Keep console clean, only INFO and above
console_handler.setFormatter(simple_formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info(f"Starting RAVEN rollout generation")
logger.info(f"Configuration: {args}")
logger.info(f"Using Azure OpenAI endpoint: {endpoint}")
logger.info(f"Model deployment: {deployment}")
logger.info(f"Log file: {log_filepath}")

# Load and process RAVEN dataset
dataset = RAVENDataset(
    data=args['prompt_path'],
    sample_max_num=args['sample_max_num'],
    sample_start_idx=args['sample_start_idx'],
)

logger.info(f"Dataset loaded: {len(dataset)} samples")

# Process all samples with optimized batch processing
batch_size = args['batch_size']
outputs = []

# Timing statistics collection
sample_times = []
total_start_time = time.time()

# Create progress bar for the full dataset - disable for screen compatibility
try:
    # Check if we're in a proper terminal
    import sys
    is_terminal = sys.stdout.isatty() and sys.stderr.isatty()
except:
    is_terminal = False

# Use progress bar only if in terminal, otherwise use simple counter
if is_terminal:
    progress_bar = tqdm(
        range(len(dataset)), 
        desc=f"Processing RAVEN samples ({args['sample_start_idx']} to {args['sample_start_idx'] + len(dataset)})",
        unit="sample"
    )
else:
    progress_bar = None
    logger.info(f"Processing {len(dataset)} RAVEN samples ({args['sample_start_idx']} to {args['sample_start_idx'] + len(dataset)})")

current_sample = 0

# Process samples in optimized batches
for batch_start in range(0, len(dataset), batch_size):
    # Check for shutdown signal
    if shutdown_flag.is_set():
        logger.info("Shutdown signal received, terminating gracefully...")
        break
        
    batch_end = min(batch_start + batch_size, len(dataset))
    batch_samples = [dataset[i] for i in range(batch_start, batch_end)]
    
    logger.info(f"Processing batch {batch_start//batch_size + 1}: samples {batch_start+1}-{batch_end}")
    batch_start_time = time.time()
    
    # Prepare batch inputs
    batch_inputs = []
    batch_items = []
    
    for i, sample in enumerate(batch_samples):
        # Check for shutdown signal during batch preparation
        if shutdown_flag.is_set():
            logger.info("Shutdown signal received during batch preparation...")
            break
            
        batch_inputs.append((sample['rollout_user_prompt'], sample['image']))
        batch_items.append(sample['item'])
    
    # Skip processing if shutdown was requested
    if shutdown_flag.is_set():
        break
    
    # Generate process supervision data for the entire batch
    curr_outputs = build_process_supervision(
        inputs=batch_inputs,
        items=batch_items,
        num_return_sequences=args['num_return_sequences'],
        args=args
    )
    
    batch_duration = time.time() - batch_start_time
    avg_sample_time = batch_duration / max(len(batch_samples), 1)  # Prevent division by zero
    sample_times.extend([avg_sample_time] * len(batch_samples))
    
    # Validate output count before extending
    expected_outputs = len(batch_samples) * args['num_return_sequences']
    actual_outputs = len(curr_outputs)
    
    if actual_outputs != expected_outputs:
        logger.error(f"Batch output count mismatch! Expected {expected_outputs}, got {actual_outputs}")
        raise ValueError(f"Batch output count mismatch: expected {expected_outputs}, got {actual_outputs}")
    
    outputs.extend(curr_outputs)
    
    # Save batch outputs incrementally
    output_file = os.path.join(args['out_dir'], f'raven_rollouts_{args["sample_start_idx"]}_{args["sample_max_num"]}.jsonl')
    file_mode = 'w' if batch_start == 0 else 'a'  # Write mode for first batch, append for subsequent
    with open(output_file, file_mode) as f:
        for output in curr_outputs:
            f.write(json.dumps(output) + '\n')
    
    logger.info(f"Saved batch {batch_start//batch_size + 1} outputs to {output_file} (total: {len(outputs)} samples)")
    
    # Update progress bar
    for i in range(len(batch_samples)):
        current_sample += 1
        sample_idx = batch_start + i
        sample = batch_samples[i]
        
        if is_terminal:
            progress_bar.update(1)
            throughput = len(batch_samples) / max(batch_duration, 0.001)  # Prevent division by zero
            progress_bar.set_postfix({
                'batch_time': f"{batch_duration:.1f}s",
                'avg_sample': f"{avg_sample_time:.1f}s",
                'throughput': f"{throughput:.2f} samples/s",
                'image': os.path.basename(sample['image_path'])[:15]
            })
        else:
            # For non-terminal environments, log progress periodically
            if current_sample % 10 == 0 or current_sample == len(dataset):
                logger.info(f"Progress: {current_sample}/{len(dataset)} samples completed ({current_sample/len(dataset)*100:.1f}%)")
                sys.stdout.flush()
                sys.stderr.flush()
        
        # Print first sample details
        if sample_idx == 0:
            logger.info(f"\n[{localtime()}] First sample processing details:")
            logger.info(f"Image path: {sample['image_path']}")
            logger.info(f"Correct answer: {sample['item']['correct_answer']}")
            logger.info(f"Batch processing time: {batch_duration:.2f} seconds")
            logger.info(f"Average sample time: {avg_sample_time:.2f} seconds")
            logger.info("\nFirst sample output:")
            print_process_supervision(curr_outputs[0])
            sys.stdout.flush()
    
    logger.info(f"Batch {batch_start//batch_size + 1} completed: {len(batch_samples)} samples in {batch_duration:.2f}s ({len(batch_samples)/batch_duration:.2f} samples/s)")
    sys.stdout.flush()
    
    # Log rate limiting stats every few batches
    if (batch_start // batch_size + 1) % 3 == 0:
        stats = rate_limiter.get_stats()
        logger.info(f"Rate Limiter Stats: RPM={stats['current_rpm']}/{stats['max_rpm']}, "
                   f"TPM={stats['current_tpm']}/{stats['max_tpm']}, "
                   f"Success Rate={stats['success_rate']:.3f} ({stats['total_requests']} total)")
        sys.stdout.flush()

if is_terminal:
    progress_bar.close()

total_end_time = time.time()
total_duration = total_end_time - total_start_time

logger.info(f"\nGenerated {len(outputs)} rollout samples")

# Final save confirmation (data already saved incrementally)
output_file = os.path.join(args['out_dir'], f'raven_rollouts_{args["sample_start_idx"]}_{args["sample_max_num"]}.jsonl')
logger.info(f"All {len(outputs)} samples saved incrementally to {output_file}")

# Print timing statistics
if sample_times:
    avg_time_per_sample = sum(sample_times) / len(sample_times)
    min_time = min(sample_times)
    max_time = max(sample_times)
    
    logger.info(f"\n=== TIMING STATISTICS ===")
    logger.info(f"Total processing time: {total_duration:.2f} seconds")
    logger.info(f"Number of samples processed: {len(sample_times)}")
    logger.info(f"Average time per sample: {avg_time_per_sample:.2f} seconds")
    logger.info(f"Minimum time per sample: {min_time:.2f} seconds") 
    logger.info(f"Maximum time per sample: {max_time:.2f} seconds")
    logger.info(f"Total time breakdown:")
    for i, duration in enumerate(sample_times):
        logger.info(f"  Sample {i+1}: {duration:.2f}s")
    
    # Estimate throughput
    samples_per_hour = 3600 / avg_time_per_sample if avg_time_per_sample > 0 else 0
    logger.info(f"\nEstimated throughput: {samples_per_hour:.2f} samples/hour")
    
    # Estimate time for full dataset
    if args['sample_max_num'] and args['sample_max_num'] < len(dataset):
        estimated_total_time = avg_time_per_sample * args['sample_max_num']
        estimated_hours = estimated_total_time / 3600
        logger.info(f"Estimated time for {args['sample_max_num']} samples: {estimated_hours:.2f} hours")

# Display summary statistics
total_steps = sum(len(output['steps_with_score']) for output in outputs)
avg_steps = total_steps / len(outputs) if outputs else 0
avg_score = sum(
    sum(step['score'] for step in output['steps_with_score']) / len(output['steps_with_score'])
    for output in outputs if output['steps_with_score']
) / len(outputs) if outputs else 0

logger.info(f"\nSummary Statistics:")
logger.info(f"Total outputs: {len(outputs)}")
logger.info(f"Average steps per output: {avg_steps:.2f}")
logger.info(f"Average step score: {avg_score:.3f}")

# Show sample output structure
if outputs:
    logger.info(f"\nSample output keys: {list(outputs[0].keys())}")
    if outputs[0]['steps_with_score']:
        logger.info(f"Sample step keys: {list(outputs[0]['steps_with_score'][0].keys())}")

logger.info("RAVEN rollout generation completed successfully")
logger.info(f"Log saved to: {log_filepath}")


# TODO: Parallelize the rollouts. 
# Based on these arguments I have set, what batch_size should I set to maximize throughput given the rate limit for the AzureOpenAI GPT-4.1 model is:

# 1. Tokens-per-minute limit: 1,000,000 tokens/min
# 2. Requests-per-minute limit: 1,000 requests/min

# Other Info:
# 1. Tokens per request: ~1,000 tokens (prompt + completion on average). can range from as low as 600 to ~2K.
# 2. Time per request: ~30s