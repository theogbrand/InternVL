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
    retry_if_exception,
    wait_random,
    before_sleep_log,
    after_log
)
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import atexit

# Initialize logger early to avoid NameError issues
logger = logging.getLogger('infovqa_rollout')

# Add the tools directory to the path
sys.path.append('/data/users/brandon/ob1-projects/InternVL/internvl_chat/tools')

from reasoning_data_pipeline.utils.accuracy_reward import (check_answer, parse_answer)

from reasoning_data_pipeline.utils.utils import localtime

# Azure OpenAI Configuration
endpoint = "https://decla-mbncunfi-australiaeast.cognitiveservices.azure.com/"
deployment = "gpt-4.1-3"
api_version = "2025-01-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=os.getenv("AZURE_API_KEY"),
    timeout=60.0,  # 60 second timeout
)

# Global shutdown flag for graceful termination
shutdown_flag = threading.Event()

def signal_handler(signum, frame):
    """Handle termination signals gracefully"""
    try:
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        # Force immediate flush for critical shutdown message only
        for handler in logger.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()
    except (NameError, AttributeError):
        # Logger not configured yet, use print as fallback
        print(f"Received signal {signum}, initiating graceful shutdown...")
        sys.stdout.flush()
    
    shutdown_flag.set()
    # Don't call sys.exit() from signal handler in threaded environment

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

class InfoVQA_Open_Ans_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        sample_start_idx=0,
        sample_end_idx=None,
    ):
        self.data = []
        total_lines = 0 # it is 1-indexed for this dataset since we are using line numbers
        
        with open(data, 'r', encoding='utf-8') as file:
            for line in file:
                total_lines += 1
                try:
                    item = json.loads(line)
                    # Include if no filtering or within range
                    if sample_end_idx is None or (sample_start_idx <= total_lines <= sample_end_idx):
                        self.data.append(line)
                        
                except json.JSONDecodeError:
                    continue

        if sample_end_idx is not None:
            print(f'Filtered {total_lines} lines to {len(self.data)} samples in range [{sample_start_idx}, {sample_end_idx}]')
        else:
            print(f'Loaded {len(self.data)} samples (no filtering)')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = json.loads(self.data[idx])
        
        # InfoVQA dataset structure: image, question, answer, uid, image_path
        image_path = item['image_path']
        question = item['question']
        answer = item['answer']
        uid = item['uid']
        
        rollout_user_prompt = r"""You are an advanced AI system specialized in Visual Question Answering (VQA) for infographics. Your objective is to examine images containing various objects, scenes, geometric shapes, diagram elements, and potentially text or numbers, and reason about processes or changes, and answer questions about their attributes, relationships, and spatial arrangements.

I will provide you with:

1. An image containing infographics
2. A question about the contents of the image

Here is the question you need to answer:

<question>
{{QUESTION}}
</question>

Please follow these steps to complete the task:

1. Carefully examine the image, paying attention to:
   - List and number all visible elements (objects, labels, arrows, text, data visualizations, etc.)
   - Determine relationships between elements
   - Interpret any text or numbers present
   - Consider the document layout and how it affects the information presentation
   - Identify any temporal or causal relationships

2. Analyze the question to identify the type of reasoning required (e.g., counting, existence check, comparison, attribute query, or relationship assessment).

3. Conduct a thorough visual analysis of the image in relation to the question, focusing on relevant elements and attributes.

4. Formulate your answer based on your analysis.

5. Present your final answer as a single string in a LaTeX-formatted box using this format: 
   <correct_answer>
   $\boxed{Your answer here}$
   </correct_answer>

Your task is to: 
- Under the [Visual Elements] section, list out all relevant visual elements step-by-step that relate to answering the question. Be thorough but concise. Wrap each step in <step_1>, <step_2>, ... tags.
- Under the [Reasoning] section, explain your step-by-step reasoning process. This should include your analysis, interpretation, and how you arrived at the answer. Provide a clear justification of how you derived the answer from the data presented. Wrap each step in <step_1>, <step_2>, ... tags.
- Present your final answer using the LaTeX-formatted box in `<correct_answer>` tags. 

It is crucial that your solution contains these sections in the exact format described below:

```
[Visual Elements]
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
$\boxed{Your answer here}$
</correct_answer>
```

Remember to:
- Provide only a single string answer in the <correct_answer> section using the $\boxed{string_answer}$ format, and no other text or commentary.""".replace('{{QUESTION}}', question)

        return {
            'rollout_user_prompt': rollout_user_prompt,
            'image': image_path,
            'image_path': image_path, # to fix: backward compatibility
            'question': question,
            'answer': answer,
            'uid': uid,
            'item': item.copy()
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
    perception_pattern = r'\[Visual Elements\](.*?)(?=\[Reasoning\]|\Z)'
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
    stop=stop_after_attempt(3),  # More attempts for rate limits
    wait=wait_exponential(multiplier=2, min=4, max=300) + wait_random(0, 30),  # Longer backoff + jitter
    retry=retry_if_exception_type((Exception,)),
    before_sleep=lambda retry_state: logger.warning(f"API retry {retry_state.attempt_number}/3: {type(retry_state.outcome.exception()).__name__}, retrying in {retry_state.next_action.sleep:.1f}s"),
    reraise=True
)
def make_azure_request(messages, max_tokens, temperature, estimated_tokens=1000):
    """Make Azure OpenAI request with retry logic"""
    try:
        response = client.chat.completions.create(
            messages=messages,
            max_completion_tokens=max_tokens,
            model=deployment,
            temperature=temperature,
            timeout=120.0
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        # Log detailed error information for monitoring
        error_type = type(e).__name__
        error_msg = str(e)
        
        # Check for content filter violation - DO NOT RETRY, return error message
        if ('BadRequestError' in error_type and 
            'Error code: 400' in error_msg and 
            ('ResponsibleAIPolicyViolation' in error_msg or 'content_filter' in error_msg)):
            logger.warning(f"Content filter violation detected, returning error response: {error_msg}")
            return "Error code 400: content filter violation returned"
        
        # Extract rate limit details if available
        if hasattr(e, 'response') and hasattr(e.response, 'headers'):
            headers = e.response.headers
            rate_limit_info = {
                'remaining_requests': headers.get('x-ratelimit-remaining-requests'),
                'remaining_tokens': headers.get('x-ratelimit-remaining-tokens'),
                'reset_requests': headers.get('x-ratelimit-reset-requests'),
                'reset_tokens': headers.get('x-ratelimit-reset-tokens')
            }
            logger.debug(f"Rate limit details: {rate_limit_info}")
        
        # Log first occurrence of error (tenacity will handle retry logging)
        if 'RateLimitError' in error_type:
            logger.debug(f"Rate limit hit: {error_msg} (estimated_tokens: {estimated_tokens})")
            # For rate limits, always retry - they're transient
            raise
        elif 'TimeoutError' in error_type or 'timeout' in error_msg.lower():
            logger.debug(f"Request timeout: {error_msg}")
            raise
        else:
            logger.debug(f"API request failed: {error_type}: {error_msg}")
            # For non-rate-limit errors, still retry but they might be permanent
            raise

def build_responses_azure_parallel(inputs, num_return_sequences=1, prefixes=None, max_new_tokens=4096, temperature=1.0, max_workers=20, args=None):
    """
    Build responses using Azure OpenAI GPT-4.1 with parallel processing and simple task-level retry
    """
    total_requests = len(inputs) * num_return_sequences
    logger.info(f"Starting parallel processing of {total_requests} requests with {max_workers} workers")
    
    def should_retry_task(exception):
        """Decide if task-level retry should happen based on exception type"""
        error_type = type(exception).__name__
        error_msg = str(exception)
        
        # Log the exception details before deciding whether to retry
        logger.debug(f"Task-level exception caught: {error_type}: {error_msg}")
        
        # Always retry rate limits at task level too for extra safety
        if 'RateLimitError' in error_type or 'rate' in error_msg.lower():
            logger.debug("Decision: RETRY (rate limit)")
            return True
        # Retry timeouts
        if 'TimeoutError' in error_type or 'timeout' in error_msg.lower():
            logger.debug("Decision: RETRY (timeout)")
            return True
        # Retry network issues
        if 'ConnectionError' in error_type or 'connection' in error_msg.lower():
            logger.debug("Decision: RETRY (connection issue)")
            return True
        # Don't retry authentication or permission errors
        if 'AuthenticationError' in error_type or 'PermissionError' in error_type:
            logger.debug("Decision: NO RETRY (auth/permission error)")
            return False
        # Retry other exceptions (could be transient)
        logger.debug("Decision: RETRY (unknown exception, assuming transient)")
        return True
    
    @retry(
        stop=stop_after_attempt(3),  # More task-level attempts
        wait=wait_exponential(multiplier=1, min=2, max=30) + wait_random(0, 10),
        retry=retry_if_exception(should_retry_task),
        before_sleep=lambda retry_state: logger.warning(f"Input {retry_state.kwargs['args_tuple'][0]}, Seq {retry_state.kwargs['args_tuple'][1]}: Task attempt {retry_state.attempt_number}/3 failed, retrying in {retry_state.next_action.sleep:.1f}s - {type(retry_state.outcome.exception()).__name__}")
    )
    def process_single_request(args_tuple):
        input_idx, seq_idx, prompt, image, prefix, args_dict = args_tuple
        
        # Check for shutdown signal
        if shutdown_flag.is_set():
            return (input_idx, seq_idx, "")
        
        try:
            # Convert image path to data URL
            try:
                data_url = local_image_to_data_url(image)
            except Exception as e:
                logger.error(f"Failed to process image for input_idx={input_idx}, seq_idx={seq_idx}: {e}")
                raise
            
            # Prepare messages
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]
            
            messages = [
                {"role": "user", "content": content}
            ]
            
            # Add prefix if provided
            if prefix:
                messages.append({"role": "assistant", "content": prefix})
            
            # Simplified token estimation
            estimated_tokens = len(prompt) // 4 + 1000 + max_new_tokens
            
            # Make the request with retries built into make_azure_request
            response_text = make_azure_request(messages, max_new_tokens, temperature, estimated_tokens)
            
            # Log response details for debugging
            if args_dict and args_dict.get('debug_granular', False) and input_idx < args_dict.get('debug_max_rollouts', 5):
                logger.debug(f"RESPONSE INPUT_{input_idx}_SEQ_{seq_idx}:\n{'='*80}\n{response_text}\n{'='*80}")
            
            return (input_idx, seq_idx, response_text)
            
        except Exception as e:
            # Log final failure and return empty string
            logger.error(f"Input {input_idx}, Seq {seq_idx}: All retry attempts exhausted - {e}")
            return (input_idx, seq_idx, "")
    
    # Prepare all request arguments
    request_args = []
    for seq_idx in range(num_return_sequences):
        for input_idx, (prompt, image) in enumerate(inputs):
            prefix = prefixes[input_idx] if prefixes else None
            request_args.append((input_idx, seq_idx, prompt, image, prefix, args))
    
    # Process requests in parallel - streamlined
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all requests
        future_to_args = {
            executor.submit(process_single_request, args): args 
            for args in request_args
        }
        
        # Collect results with minimal overhead
        for future in as_completed(future_to_args.keys(), timeout=3600):
            try:
                input_idx, seq_idx, response = future.result(timeout=600)
                results[(input_idx, seq_idx)] = response
            except Exception as e:
                args = future_to_args[future]
                logger.error(f"Future failed for input {args[0]}, seq {args[1]}: {e}")
                results[(args[0], args[1])] = ""
    
    # Reconstruct response list in correct order
    response_list = []
    for input_idx in range(len(inputs)):
        for seq_idx in range(num_return_sequences):
            response = results.get((input_idx, seq_idx), "")
            response_list.append(response)
    
    successful_requests = sum(1 for r in response_list if r != "")
    logger.info(f"Completed parallel processing: {successful_requests}/{len(response_list)} responses generated successfully")
    return response_list

# Updated function call
def build_responses_azure(inputs, num_return_sequences=1, prefixes=None, max_new_tokens=4096, temperature=1.0, max_workers=None, args=None):
    """Wrapper for backward compatibility with explicit max_workers parameter"""
    # Use provided max_workers or default to 20
    if max_workers is None:
        max_workers = args.get('max_workers', 20) if args else 20
    return build_responses_azure_parallel(
        inputs, num_return_sequences, prefixes, max_new_tokens, temperature, max_workers, args
    )


def build_mc_scores_maximum_throughput(inputs, response_list, items, num_return_sequences, args):
    """
    STREAMING PIPELINE: 1K rollouts → M MC tasks → time-based processing → per-rollout completion tracking
    """
    assert len(response_list) == len(inputs) * num_return_sequences
    
    logger.info(f"Starting STREAMING MC pipeline for {len(response_list)} rollouts")
    
    # Step 1: Parse all rollouts and create MC task queue with completion tracking
    mc_task_queue = []
    rollout_metadata = {}  # rollout_idx -> {steps, total_mc_tasks, completed_mc_tasks, results}
    
    parsing_stats = {'success': 0, 'failures': 0}
    
    for rollout_idx, response in enumerate(response_list):
        input_data = inputs[rollout_idx // num_return_sequences]
        item = items[rollout_idx // num_return_sequences]
        
        try:
            # Parse the response into steps
            steps = parse_response_to_perception_and_reasoning_steps_and_correct_answer(
                response, 
                max_perception_steps=args.get('max_perception_steps', 12), 
                max_reasoning_steps=args.get('max_reasoning_steps', 12)
            )
            
            # Combine perception and reasoning steps
            flat_steps = steps['perception_steps'] + steps['reasoning_steps']
            perception_count = len(steps['perception_steps'])
            
            # Initialize rollout tracking
            num_mc_per_step = args.get('num_mc_sequences', 8)
            total_mc_tasks = len(flat_steps) * num_mc_per_step
            
            rollout_metadata[rollout_idx] = {
                'input_data': input_data,
                'item': item,
                'response': response,
                'steps': flat_steps,
                'perception_count': perception_count,
                'total_mc_tasks': total_mc_tasks,
                'completed_mc_tasks': 0,
                'mc_results': {},  # (step_idx, mc_idx) -> result
                'is_complete': False
            }
            
            # Create MC tasks for this rollout
            for step_idx in range(len(flat_steps)):
                # Build formatted prefix for this step
                prefix_steps = flat_steps[:step_idx+1]
                
                formatted_prefix = ""
                if step_idx < perception_count:
                    formatted_prefix += "[Visual Elements]\n"
                    for i, step in enumerate(prefix_steps):
                        formatted_prefix += f"<step_{i+1}>\n{step}\n</step_{i+1}>\n"
                else:
                    formatted_prefix += "[Visual Elements]\n"
                    for i, step in enumerate(steps['perception_steps']):
                        formatted_prefix += f"<step_{i+1}>\n{step}\n</step_{i+1}>\n"
                    formatted_prefix += "\n[Reasoning]\n"
                    reasoning_steps = prefix_steps[perception_count:]
                    for i, step in enumerate(reasoning_steps):
                        formatted_prefix += f"<step_{i+1}>\n{step}\n</step_{i+1}>\n"
                
                prefix = formatted_prefix.strip()
                
                # Debug log for first few rollouts
                if args.get('debug_granular', False) and rollout_idx < args.get('debug_max_rollouts', 5):
                    logger.debug(f"ROLLOUT {rollout_idx} STEP {step_idx}: Step content: {flat_steps[step_idx]}")
                    logger.debug(f"ROLLOUT {rollout_idx} STEP {step_idx}: Prepared prefix {prefix}")
                
                # Create MC tasks for this step
                for mc_idx in range(num_mc_per_step):
                    task_id = f"R{rollout_idx}-S{step_idx}-MC{mc_idx}"
                    mc_task_queue.append({
                        'rollout_idx': rollout_idx,
                        'step_idx': step_idx,
                        'mc_idx': mc_idx,
                        'input_data': input_data,
                        'prefix': prefix,
                        'answer_gt': item['answer'],
                        'task_id': task_id
                    })
                    
                    # Debug log for first few MC tasks
                    if args.get('debug_granular', False) and rollout_idx < args.get('debug_max_rollouts', 5) and mc_idx == 0:
                        logger.debug(f"MC TASK {task_id}: GT Answer = {item['answer']}")
            
            parsing_stats['success'] += 1
            if rollout_idx < 3:  # Log first few successfully parsed rollouts
                logger.info(f"✓ Rollout {rollout_idx} parsed successfully: {len(flat_steps)} steps, {total_mc_tasks} MC tasks")
                logger.debug(f"  Rollout {rollout_idx} metadata: total_mc_tasks={rollout_metadata[rollout_idx]['total_mc_tasks']}, completed_mc_tasks={rollout_metadata[rollout_idx]['completed_mc_tasks']}")
                    
        except Exception as e:
            logger.error(f"✗ Failed to parse rollout {rollout_idx}: {e}")
            if rollout_idx < 5:  # Show first few parsing failures in detail
                try:
                    logger.error("=" * 80)
                    logger.error("Failed response text:")
                    logger.error("-" * 40)
                    if isinstance(response, str):
                        logger.error(response)
                    else:
                        logger.error(f"Response is not a string: {type(response)}")
                        logger.error(str(response))
                    logger.error("-" * 40)
                    logger.error("=" * 80)
                except Exception as log_error:
                    logger.error(f"Failed to log response text: {log_error}")
            # Create minimal tracking for failed rollout
            rollout_metadata[rollout_idx] = {
                'input_data': input_data,
                'item': item,
                'response': response,
                'steps': ['Error parsing'],
                'perception_count': 0,
                'total_mc_tasks': 1,
                'completed_mc_tasks': 1,  # Mark as complete
                'mc_results': {(0, 0): ""},
                'is_complete': False  # Allow saving in completion logic
            }
            parsing_stats['failures'] += 1
    
    total_mc_tasks = len(mc_task_queue)
    total_rollouts = len(response_list)
    avg_mc_per_rollout = total_mc_tasks / total_rollouts
    
    logger.info(f"PARSING STATISTICS:")
    logger.info(f"  ✓ Successfully parsed: {parsing_stats['success']}/{total_rollouts} rollouts ({parsing_stats['success']/total_rollouts*100:.1f}%)")
    logger.info(f"  ✗ Failed to parse: {parsing_stats['failures']}/{total_rollouts} rollouts ({parsing_stats['failures']/total_rollouts*100:.1f}%)")
    
    logger.info(f"MC Task Queue Created:")
    logger.info(f"  Total rollouts: {total_rollouts}")
    logger.info(f"  Total MC tasks: {total_mc_tasks}")
    logger.info(f"  Average MC tasks per rollout: {avg_mc_per_rollout:.1f}")
    logger.info(f"  Estimated batches: {math.ceil(total_mc_tasks/900)}")
    logger.info(f"  Estimated time: {math.ceil(total_mc_tasks/900)*60:.0f} seconds")
    
    # Step 2: Process MC tasks with time-based firing + streaming completion tracking
    throughput_batch_size = 1900  # adjust so each batch takes about 1 minute (which maximizes the 1M TPM)
    all_batches = [mc_task_queue[i:i+throughput_batch_size] for i in range(0, total_mc_tasks, throughput_batch_size)]
    
    # Output file for streaming saves
    output_file = os.path.join(args['out_dir'], f'{args["dataset_name"]}_raven_rollouts_{args["sample_start_idx"]}_{args["sample_end_idx"]}_streaming.jsonl')
    completed_rollouts = 0
    
    logger.info(f"Starting time-based MC processing with streaming saves ({throughput_batch_size} RPM rate limit)")
    start_time = time.time()
    
    # Open output file immediately for true streaming (append mode to preserve existing rollouts)
    with open(output_file, 'a', encoding='utf-8') as f:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            
            # Fire batches every 60 seconds to maintain throughput_batch_size RPM throughput
            for batch_idx, batch_tasks in enumerate(all_batches):
                fire_time = start_time + batch_idx * 60.0
                current_time = time.time()
                
                if current_time < fire_time:
                    wait_time = fire_time - current_time
                    logger.info(f"Waiting {wait_time:.1f}s to fire batch {batch_idx + 1}/{len(all_batches)}")
                    time.sleep(wait_time)
                
                logger.info(f"Firing batch {batch_idx + 1}/{len(all_batches)}: {len(batch_tasks)} MC requests")
                future = executor.submit(process_mc_batch_simple, batch_tasks, batch_idx, args)
                futures.append((future, batch_tasks))
            
            # Process results AS THEY COMPLETE (not sequentially) for true streaming
            from concurrent.futures import as_completed
            
            future_to_tasks = {future: batch_tasks for future, batch_tasks in futures}
            
            for future in as_completed(future_to_tasks.keys()):
                batch_tasks = future_to_tasks[future]
                try:
                    batch_responses = future.result()
                    
                    # Debug: Log which batch completed
                    batch_rollout_indices = list(set(task['rollout_idx'] for task in batch_tasks))
                    logger.info(f"Processing completed batch with {len(batch_tasks)} tasks affecting rollouts {min(batch_rollout_indices)}-{max(batch_rollout_indices)}")
                    
                    # Process each MC result and update rollout tracking
                    for task, mc_response in zip(batch_tasks, batch_responses):
                        rollout_idx = task['rollout_idx']
                        step_idx = task['step_idx']
                        mc_idx = task['mc_idx']
                        task_id = task['task_id']
                        
                        # Debug log MC response for first few rollouts
                        if args.get('debug_granular', False) and rollout_idx < args.get('debug_max_rollouts', 5):
                            logger.debug(f"MC RESPONSE {task_id}: {mc_response[:200]}..." if len(mc_response) > 200 else f"MC RESPONSE {task_id}: {mc_response}")
                        
                        # Store MC result
                        rollout_metadata[rollout_idx]['mc_results'][(step_idx, mc_idx)] = mc_response
                        rollout_metadata[rollout_idx]['completed_mc_tasks'] += 1
                        
                        # Check if this rollout is now complete (SUCCESS CASE)
                        rollout_meta = rollout_metadata[rollout_idx]
                        
                        # Debug logging for first few rollouts
                        if rollout_idx < 5:
                            logger.debug(f"Rollout {rollout_idx}: {rollout_meta['completed_mc_tasks']}/{rollout_meta['total_mc_tasks']} MC tasks, is_complete={rollout_meta['is_complete']}")
                        
                        if (rollout_meta['completed_mc_tasks'] >= rollout_meta['total_mc_tasks'] and 
                            not rollout_meta['is_complete']):
                            
                            # Mark complete and save immediately
                            rollout_meta['is_complete'] = True
                            completed_rollouts += 1
                            
                            # Build final output for this rollout
                            rollout_output = build_rollout_output(rollout_idx, rollout_meta, args)
                            
                            # Save to JSONL immediately
                            f.write(json.dumps(rollout_output, ensure_ascii=False) + '\n')
                            f.flush()  # Ensure immediate write
                            
                            logger.info(f"Rollout {rollout_idx} completed and saved ({completed_rollouts}/{total_rollouts})")
                    
                    progress = sum(meta['completed_mc_tasks'] for meta in rollout_metadata.values())
                    logger.info(f"Batch completed: {progress}/{total_mc_tasks} MC tasks ({progress/total_mc_tasks*100:.1f}%)")
                    
                except Exception as e:
                    logger.error(f"Batch failed: {e}")
                    # Mark affected rollouts as complete with dummy results
                    for task in batch_tasks:
                        rollout_idx = task['rollout_idx']
                        rollout_metadata[rollout_idx]['mc_results'][(task['step_idx'], task['mc_idx'])] = ""
                        rollout_metadata[rollout_idx]['completed_mc_tasks'] += 1
                        
                        # Check if this rollout is now complete due to failure
                        rollout_meta = rollout_metadata[rollout_idx]
                        if (rollout_meta['completed_mc_tasks'] >= rollout_meta['total_mc_tasks'] and 
                            not rollout_meta['is_complete']):
                            
                            # Mark complete and save immediately
                            rollout_meta['is_complete'] = True
                            completed_rollouts += 1
                            
                            # Build final output for this rollout
                            rollout_output = build_rollout_output(rollout_idx, rollout_meta, args)
                            
                            # Save to JSONL immediately
                            f.write(json.dumps(rollout_output, ensure_ascii=False) + '\n')
                            f.flush()  # Ensure immediate write
                            
                            logger.info(f"Rollout {rollout_idx} completed and saved (from failed batch) ({completed_rollouts}/{total_rollouts})")
    
    total_duration = time.time() - start_time
    actual_rate = total_mc_tasks / total_duration * 60
    
    logger.info(f"STREAMING MC Pipeline Completed:")
    logger.info(f"  Total MC tasks: {total_mc_tasks}")
    logger.info(f"  Completed rollouts: {completed_rollouts}/{total_rollouts}")
    logger.info(f"  Duration: {total_duration:.0f}s ({actual_rate:.0f} tasks/min, target: {throughput_batch_size} RPM)")
    logger.info(f"  Saved to: {output_file}")
    
    # Return empty list since we saved incrementally
    return []

def build_rollout_output(rollout_idx, rollout_meta, args):
    """Build final output structure for a completed rollout"""
    steps_with_score = []
    num_mc_sequences = args.get('num_mc_sequences', 8)
    
    # Debug log for first few rollouts
    if args.get('debug_granular', False) and rollout_idx < args.get('debug_max_rollouts', 5):
        logger.debug(f"BUILDING OUTPUT FOR ROLLOUT {rollout_idx}:")
        logger.debug(f"  Total steps: {len(rollout_meta['steps'])}")
        logger.debug(f"  GT Answer: {rollout_meta['item']['answer']}")
    
    for step_idx in range(len(rollout_meta['steps'])):
        # Collect MC results for this step
        mc_correctness = []
        mc_details = []  # For debug logging
        
        for mc_idx in range(num_mc_sequences):
            mc_response = rollout_meta['mc_results'].get((step_idx, mc_idx), "")
            
            try:
                parsed_answer = parse_answer(mc_response, prompt_version=args.get('prompt_format_version', ''))
                answer_pred = parsed_answer[-1]
                correctness = check_answer(
                    answer_pred=answer_pred,
                    answer_gt=str(rollout_meta['item']['answer']),
                    mode=args.get('scoring_mode', ''),
                    image_path=rollout_meta['item']['image_path'],
                    question=rollout_meta['item']['question']
                )
                mc_details.append(f"MC{mc_idx}: {answer_pred} -> {correctness}")
            except Exception as e:
                correctness = 0
                mc_details.append(f"MC{mc_idx}: PARSE_ERROR -> 0")
                if args.get('debug_granular', False) and rollout_idx < args.get('debug_max_rollouts', 5):
                    logger.debug(f"ROLLOUT {rollout_idx} STEP {step_idx} MC{mc_idx}: Parse error - {e}")
            
            mc_correctness.append(correctness)
        
        # Calculate step score
        score = sum(mc_correctness) / len(mc_correctness) if mc_correctness else 0.0
        
        # Debug log for first few rollouts
        if args.get('debug_granular', False) and rollout_idx < args.get('debug_max_rollouts', 5):
            logger.debug(f"ROLLOUT {rollout_idx} STEP {step_idx}: Score = {score:.3f} ({sum(mc_correctness)}/{len(mc_correctness)})")
            logger.debug(f"  MC Details: {' | '.join(mc_details)}")
        
        step_output = {
            'step': rollout_meta['steps'][step_idx],
            'score': score,
            'num_mc_correct': sum(mc_correctness),
            'num_mc_total': len(mc_correctness),
        }
        
        steps_with_score.append(step_output)
        
        # Apply early stopping
        if args.get('early_stop', True) and score == 0.0:
            if args.get('debug_granular', False) and rollout_idx < args.get('debug_max_rollouts', 5):
                logger.debug(f"ROLLOUT {rollout_idx}: Early stopping at step {step_idx} (score = 0.0)")
            break
    
    # Build final output
    output = rollout_meta['item'].copy()
    output['response'] = rollout_meta['response']
    output['steps_with_score'] = steps_with_score
    output['question'] = rollout_meta['input_data'][0]
    
    # Add parsing_failed flag for failed rollouts (for retry capability)
    if rollout_meta.get('steps') == ['Error parsing']:
        output['parsing_failed'] = True
    # Successfully parsed rollouts do NOT get this key at all
    
    # Debug log final output for first few rollouts
    if args.get('debug_granular', False) and rollout_idx < args.get('debug_max_rollouts', 5):
        avg_score = sum(step['score'] for step in steps_with_score) / len(steps_with_score) if steps_with_score else 0.0
        parsing_status = "FAILED" if output.get('parsing_failed') else "SUCCESS"
        logger.debug(f"ROLLOUT {rollout_idx} FINAL: {len(steps_with_score)} steps, avg_score = {avg_score:.3f}, parsing = {parsing_status}")
    
    return output

def process_mc_batch_simple(batch_tasks, batch_idx, args):
    """
    Process a single batch of MC tasks with simple time-based batch firing
    Each batch contains up to 1000 independent MC evaluation requests
    """
    logger.info(f"Starting batch {batch_idx + 1}: {len(batch_tasks)} MC tasks")
    batch_start_time = time.time()
    
    # Debug log for first few tasks in early batches
    if args.get('debug_granular', False) and batch_idx < 2:
        for i, task in enumerate(batch_tasks[:3]):  # Log first 3 tasks
            logger.debug(f"BATCH {batch_idx + 1} TASK {i}: {task['task_id']}")
            logger.debug(f"  Prefix length: {len(task['prefix'])}")
            logger.debug(f"  GT Answer: {task['answer_gt']}")
    
    # Prepare batch inputs
    batch_inputs = [task['input_data'] for task in batch_tasks]
    batch_prefixes = [task['prefix'] for task in batch_tasks]
    
    # Process batch - each task gets 1 API call
    batch_responses = build_responses_azure(
        batch_inputs,
        1,  # Single response per task (not multiple MC per input)
        batch_prefixes,
        max_new_tokens=args.get('max_new_tokens', 4096),
        temperature=args.get('temperature', 1.0),
        max_workers=args.get('max_workers', 80),
        args=args
    )
    
    batch_duration = time.time() - batch_start_time
    actual_rate = len(batch_tasks) / batch_duration * 60  # Convert to per minute
    
    logger.info(f"Batch {batch_idx + 1} processing completed: "
               f"{len(batch_tasks)} tasks in {batch_duration:.1f}s "
               f"({actual_rate:.0f} tasks/min)")
    
    return batch_responses

def build_process_supervision(inputs, items, num_return_sequences, args):
    """
    STREAMING PIPELINE: 1K initial rollouts → M MC tasks → time-based processing → incremental saves
    """
    total_inputs = len(inputs)
    total_initial_requests = total_inputs * num_return_sequences
    
    logger.info(f"=== STREAMING ROLLOUT PIPELINE ===")
    logger.info(f"Phase 1 - Initial Rollout Generation:")
    logger.info(f"  Inputs: {total_inputs}")
    logger.info(f"  Sequences per input: {num_return_sequences}")
    logger.info(f"  Total requests: {total_initial_requests}")
    logger.info(f"  Rate limit utilization: {total_initial_requests/500*100:.1f}% of 500 RPM")
    
    initial_start_time = time.time()
    
    response_list = build_responses_azure(
        inputs, # rollout_user_prompt, image
        num_return_sequences,
        max_new_tokens=args.get('max_new_tokens', 4096),
        temperature=args.get('temperature', 1.0),
        max_workers=args.get('max_workers', 80),
        args=args
    )
    
    initial_duration = time.time() - initial_start_time
    actual_initial_rate = total_initial_requests / initial_duration * 60
    
    logger.info(f"Phase 1 completed:")
    logger.info(f"  Duration: {initial_duration:.2f} seconds")
    logger.info(f"  Actual rate: {actual_initial_rate:.1f} RPM ({actual_initial_rate/500*100:.1f}% of limit)")
    logger.info(f"  Generated {len(response_list)} rollout responses")

    logger.info(f"\nPhase 2 - Streaming MC Pipeline:")
    logger.info(f"All MC tasks processed with time-based firing + per-rollout completion tracking")
    
    # Process with streaming pipeline (saves incrementally, returns empty)
    build_mc_scores_maximum_throughput(inputs, response_list, items, num_return_sequences, args)

    total_duration = time.time() - initial_start_time
    logger.info(f"\n=== STREAMING PIPELINE SUMMARY ===")
    logger.info(f"Total processing time: {total_duration:.2f} seconds")
    logger.info(f"All rollouts saved incrementally as they completed")
    logger.info(f"Ready for next batch of {total_inputs} inputs")

    # Return empty since we saved incrementally
    return []


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


# Configuration parameters optimized for MAXIMUM throughput in BOTH phases
args = {
    'endpoint': endpoint,
    'deployment': deployment,
    'api_version': api_version,
    'prompt_path': '/data/users/brandon/ob1-projects/InternVL/internvl_chat/rollout_generation/preprocessed_prompts/preprocessing_scripts/InfoVQA/prepared_jsonl/infovqa_run1_open_ans_9K_v1_subset.jsonl',
    'out_dir': 'infovqa_open_answer_rollouts_output',
    'batch_size': 15,  # ~20 samples per batch
    'num_return_sequences': 4,  # 20×4 = 80 requests per batch (ensure this is FAST less than 20s so we are rate limited at the TPM level in phase 2)
    'sample_start_idx': 645,
    'sample_end_idx': 966,
    'prompt_format_version': 'dvqa_v1_int_only', # reuse boxed answer format, and open ended scoring handled by ai2d 
    'scoring_mode': 'ai2d_open_answer_score', # reuse for open ans
    'num_mc_sequences': 16,  # 16 MC sequences per rollout
    'max_perception_steps': 12,
    'max_reasoning_steps': 12,
    'early_stop': False, # when a step results in an incorrect answer we immediately stop rollouts from THAT step.
    'max_new_tokens': 4096,
    'temperature': 1.0,
    'max_workers': (os.cpu_count() or 4) * 8,  # 8x CPU cores for I/O-bound API calls
    'debug_granular': True,  # Enable granular rollout-level debug logging
    'debug_max_rollouts': 1,  # Limit detailed logging to first N rollouts per batch to avoid log bloat
}

# Extract dataset name from prompt_path for unique file naming
dataset_name = os.path.splitext(os.path.basename(args['prompt_path']))[0]
args['dataset_name'] = dataset_name

def main():
    """Main execution function for RAVEN rollout generation"""
    # Create output directory
    os.makedirs(args['out_dir'], exist_ok=True)

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logs_dir = os.path.join(args['out_dir'], f"{args['dataset_name']}_rollout_logs")
    os.makedirs(logs_dir, exist_ok=True)

    log_filename = f"{args['dataset_name']}_rollout_{timestamp}_samples_{args['sample_start_idx']}_{args['sample_end_idx']}.txt"
    log_filepath = os.path.join(logs_dir, log_filename)

    # Configure the logger (already created at top of file)
    logger.setLevel(logging.DEBUG)  # Allow all levels to reach handlers

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter('%(message)s')

    # File handler - detailed logging (includes DEBUG)
    # Add basic log rotation to prevent disk issues
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        log_filepath,
        maxBytes=100*1024*1024,  # 100MB per file
        backupCount=2,  # Keep 2 backup files
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)  # Save all debug info to file
    file_handler.setFormatter(detailed_formatter)

    # Console handler - replace TqdmLoggingHandler with StreamHandler for screen compatibility
    console_handler = logging.StreamHandler(sys.stdout)
    # Show debug messages on console if granular debugging is enabled
    console_level = logging.DEBUG if args.get('debug_granular', False) else logging.INFO
    console_handler.setLevel(console_level)
    console_handler.setFormatter(simple_formatter)

    # Add handlers to logger (prevent duplicates)
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    logger.info(f"Starting RAVEN rollout generation")
    logger.info(f"Configuration: {args}")
    logger.info(f"Using Azure OpenAI endpoint: {endpoint}")
    logger.info(f"Model deployment: {deployment}")
    logger.info(f"Log file: {log_filepath}")

    atexit.register(lambda: logging.shutdown())

    # Load and process RAVEN dataset
    dataset = InfoVQA_Open_Ans_Dataset(
        data=args['prompt_path'],
        sample_start_idx=args['sample_start_idx'],
        sample_end_idx=args['sample_end_idx'],
    )

    logger.info(f"Dataset loaded: {len(dataset)} samples")

    # Process all samples with optimized batch processing
    batch_size = args['batch_size']

    # Timing statistics collection
    sample_times = []
    total_start_time = time.time()

    # Create progress bar for the full dataset - disable for screen compatibility
    try:
        # Check if we're in a proper terminal
        is_terminal = sys.stdout.isatty() and sys.stderr.isatty()
    except:
        is_terminal = False

    # Use progress bar only if in terminal, otherwise use simple counter
    if is_terminal:
        progress_bar = tqdm(
            range(len(dataset)), 
            desc=f"Processing {args['dataset_name']} samples (ID range {args['sample_start_idx']} to {args['sample_end_idx']})",
            unit="sample"
        )
    else:
        progress_bar = None
        logger.info(f"Processing {len(dataset)} {args['dataset_name']} samples (ID range {args['sample_start_idx']} to {args['sample_end_idx']})")

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
        logger.info(f"OPTIMIZED THROUGHPUT: {len(batch_samples)} samples × {args['num_return_sequences']} sequences = {len(batch_samples) * args['num_return_sequences']} initial requests")
        logger.info(f"Rate limit utilization: {len(batch_samples) * args['num_return_sequences']/500*100:.1f}% of 500 RPM limit")
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
        
        # Streaming pipeline handles validation internally
        
        # curr_outputs is empty since streaming pipeline saves incrementally
        # No need to save here as rollouts are saved as they complete
        
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
                logger.info(f"Correct answer: {sample['item']['answer']}")
                logger.info(f"Batch processing time: {batch_duration:.2f} seconds")
                logger.info(f"Average sample time: {avg_sample_time:.2f} seconds")
                logger.info(f"Outputs saved incrementally via streaming pipeline")
                sys.stdout.flush()
        
        logger.info(f"Batch {batch_start//batch_size + 1} completed: {len(batch_samples)} samples in {batch_duration:.2f}s ({len(batch_samples)/batch_duration:.2f} samples/s)")
        
        # Debug summary for this batch if granular debugging is enabled
        if args.get('debug_granular', False):
            logger.debug(f"BATCH {batch_start//batch_size + 1} DEBUG SUMMARY:")
            logger.debug(f"  Samples processed: {len(batch_samples)}")
            logger.debug(f"  Expected rollouts: {len(batch_samples) * args['num_return_sequences']}")
            logger.debug(f"  All rollouts saved via streaming pipeline")
        
        sys.stdout.flush()
        
        # Log progress periodically
        if (batch_start // batch_size + 1) % 3 == 0:
            sys.stdout.flush()

    if is_terminal:
        progress_bar.close()

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    logger.info(f"\nProcessed {len(dataset)} input samples via streaming pipeline")

    # Final save confirmation (data already saved incrementally via streaming)
    output_file = os.path.join(args['out_dir'], f'{args["dataset_name"]}_raven_rollouts_id_{args["sample_start_idx"]}_to_{args["sample_end_idx"]}_streaming.jsonl')
    logger.info(f"All rollouts saved incrementally to {output_file}")

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
        if args['sample_end_idx'] and args['sample_end_idx'] < len(dataset):
            estimated_total_time = avg_time_per_sample * (args['sample_end_idx'] - args['sample_start_idx'])
            estimated_hours = estimated_total_time / 3600
            logger.info(f"Estimated time for {args['sample_end_idx'] - args['sample_start_idx']} samples: {estimated_hours:.2f} hours")

    # Summary statistics available in streaming output file
    logger.info(f"\nSummary Statistics:")
    logger.info(f"Total input samples processed: {len(dataset)}")
    logger.info(f"Expected total rollouts: {len(dataset) * args['num_return_sequences']}")
    logger.info(f"All rollout statistics available in output file")

    logger.info("RAVEN rollout generation completed successfully with OPTIMIZED THROUGHPUT")
    logger.info(f"✓ Phase 1: Conservative RPM utilization for initial rollouts")
    logger.info(f"✓ Phase 2: All MC tasks processed independently with optimized parallelization")
    logger.info(f"✓ Both phases optimized for 500 RPM sustained throughput")
    logger.info(f"Log saved to: {log_filepath}")

if __name__ == "__main__":
    main()