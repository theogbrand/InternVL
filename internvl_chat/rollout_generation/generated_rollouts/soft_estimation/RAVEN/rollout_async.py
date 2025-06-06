import os
import sys
import json
import math
import base64
import time
import asyncio
import aiofiles
from mimetypes import guess_type
from collections import defaultdict
import torch
from PIL import Image
from openai import AsyncAzureOpenAI

# Add the tools directory to the path
sys.path.append('/data/users/brandon/ob1-projects/InternVL/internvl_chat/tools')

from reasoning_data_pipeline.utils.accuracy_reward import (check_answer, parse_answer)
from reasoning_data_pipeline.utils.utils import localtime

# Azure OpenAI Configuration
endpoint = "https://dalle-declare.openai.azure.com/"
deployment = "gpt-4.1"
api_version = "2025-01-01-preview"

async_client = AsyncAzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=os.getenv("AZURE_API_KEY"),
)

class RateLimiter:
    def __init__(self, max_requests_per_minute=950, max_tokens_per_minute=950000):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        
        # Track requests and tokens in sliding windows
        self.request_times = []
        self.token_usage = []  # (timestamp, tokens)
        
        # Semaphore to limit concurrent requests
        self.request_semaphore = asyncio.Semaphore(50)  # Max 50 concurrent
        
    async def acquire(self, estimated_tokens=1000):
        async with self.request_semaphore:
            now = time.time()
            
            # Clean old entries (older than 1 minute)
            self.request_times = [t for t in self.request_times if now - t < 60]
            self.token_usage = [(t, tokens) for t, tokens in self.token_usage if now - t < 60]
            
            # Check if we need to wait
            current_requests = len(self.request_times)
            current_tokens = sum(tokens for _, tokens in self.token_usage)
            
            # Calculate wait time if needed
            wait_time = 0
            if current_requests >= self.max_requests_per_minute:
                wait_time = max(wait_time, 60 - (now - self.request_times[0]))
            
            if current_tokens + estimated_tokens >= self.max_tokens_per_minute:
                oldest_token_time = self.token_usage[0][0] if self.token_usage else now
                wait_time = max(wait_time, 60 - (now - oldest_token_time))
            
            if wait_time > 0:
                print(f"Rate limiting: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
            
            # Record this request
            self.request_times.append(now)
            self.token_usage.append((now, estimated_tokens))

# Global rate limiter
rate_limiter = RateLimiter()

# Function to encode a local image into data URL 
def local_image_to_data_url(image_path):
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'

    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    return f"data:{mime_type};base64,{base64_encoded_data}"

async def make_azure_request(prompt, image, prefix=None, max_new_tokens=4096, temperature=1.0):
    """Make a single async request to Azure OpenAI"""
    try:
        # Rate limiting
        await rate_limiter.acquire(estimated_tokens=1000)
        
        # Convert image to data URL
        if isinstance(image, str):
            data_url = local_image_to_data_url(image)
        else:
            temp_path = f"/tmp/temp_image_{asyncio.current_task().get_name()}.png"
            image.save(temp_path)
            data_url = local_image_to_data_url(temp_path)
            os.remove(temp_path)
        
        # Prepare messages
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}}
        ]
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that excels at visual reasoning and pattern recognition."},
            {"role": "user", "content": content}
        ]
        
        if prefix is not None:
            messages.append({"role": "assistant", "content": prefix})
        
        # Call Azure OpenAI
        response = await async_client.chat.completions.create(
            messages=messages,
            max_completion_tokens=max_new_tokens,
            model=deployment,
            temperature=temperature
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error in Azure request: {e}")
        return ""

async def build_responses_azure_async(inputs, num_return_sequences=1, prefixes=None, max_new_tokens=4096, temperature=1.0):
    """Build responses using async Azure OpenAI with parallelization"""
    tasks = []
    
    for seq_idx in range(num_return_sequences):
        for input_idx, (prompt, image) in enumerate(inputs):
            prefix = prefixes[input_idx] if prefixes is not None else None
            
            task = asyncio.create_task(
                make_azure_request(prompt, image, prefix, max_new_tokens, temperature),
                name=f"req_{input_idx}_{seq_idx}"
            )
            tasks.append((input_idx, seq_idx, task))
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*[task for _, _, task in tasks], return_exceptions=True)
    
    # Organize results
    batched_response_list = [[] for _ in range(len(inputs))]
    for (input_idx, seq_idx, _), result in zip(tasks, results):
        if isinstance(result, Exception):
            print(f"Task failed for input {input_idx}, seq {seq_idx}: {result}")
            result = ""
        batched_response_list[input_idx].append(result)
    
    return sum(batched_response_list, start=[])

class RAVENDataset(torch.utils.data.Dataset):
    def __init__(self, data, sample_max_num=None, sample_start_idx=0):
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
        image_path = item['combined_image_path']
        correct_answer = item['correct_answer']
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
    """Parse text that contains perception steps, reasoning steps, and a correct answer."""
    import re
    
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
    
    reasoning_steps.sort(key=lambda x: int(x[0]))
    result['reasoning_steps'] = [step[1].strip() for step in reasoning_steps]
    
    # Extract correct answer
    answer_pattern = r'<correct_answer>(.*?)</correct_answer>'
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    
    if not answer_match:
        raise ValueError("Could not find correct answer")
    
    result['llm_answer'] = answer_match.group(1).strip()
    
    if not result['perception_steps'] or not result['reasoning_steps'] or not result['llm_answer']:
        raise ValueError("Missing one or more required components")
    
    return result

async def build_mc_scores_async(inputs, response_list, items, num_return_sequences, args):
    """Build Monte Carlo scores with async processing"""
    assert len(response_list) == len(inputs) * num_return_sequences

    steps_list = []
    for response in response_list:
        try:
            steps = parse_response_to_perception_and_reasoning_steps_and_correct_answer(
                response, 
                max_perception_steps=args.get('max_perception_steps', 12), 
                max_reasoning_steps=args.get('max_reasoning_steps', 12)
            )
            steps_list.append(steps)
        except Exception as e:
            print(f"Failed to parse response: {e}")
            steps_list.append({'perception_steps': ['Error parsing'], 'reasoning_steps': ['Error parsing'], 'llm_answer': 'Error'})
    
    # Convert to flat steps
    flat_steps_list = []
    for steps_dict in steps_list:
        flat_steps = steps_dict['perception_steps'] + steps_dict['reasoning_steps']
        flat_steps_list.append(flat_steps)
    
    steps_flag = [False for _ in range(len(response_list))]
    steps_outputs = [[] for _ in range(len(response_list))]

    step_cnt = 0
    while True:
        print(f"=== STEP_CNT = {step_cnt} ===")
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
            
            input = inputs[idx // num_return_sequences]
            item = items[idx // num_return_sequences]

            curr_inputs_idx.append(idx)
            curr_inputs.append(input)
            
            # Build prefix
            prefix_steps = flat_steps[:step_cnt+1]
            perception_count = len(steps_list[idx]['perception_steps'])
            
            if step_cnt < perception_count:
                perception_prefix = prefix_steps
                reasoning_prefix = []
            else:
                perception_prefix = steps_list[idx]['perception_steps']
                reasoning_prefix = prefix_steps[perception_count:]
            
            formatted_prefix = ""
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
            curr_prefixes.append(formatted_prefix.strip())

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

        # Async MC evaluation - this is where the speedup happens
        mc_response_list = await build_responses_azure_async(
            curr_inputs, 
            args.get('num_mc_sequences', 16), 
            curr_prefixes,
            max_new_tokens=args.get('max_new_tokens', 4096),
            temperature=args.get('temperature', 1.0)
        )

        # Check correctness
        correctness_list = []
        for mc_idx, mc_response in enumerate(mc_response_list):
            try:
                correctness = check_answer(
                    answer_pred=parse_answer(mc_response, prompt_version=args.get('prompt_version', 'raven_v1'))[-1],
                    answer_gt=str(curr_answer_gt[mc_idx // args.get('num_mc_sequences', 16)]),
                    mode='raven_score'
                )
            except Exception as e:
                print(f'Fail to check correctness for response: {mc_response[:100]}... Error: {e}')
                correctness = 0
            correctness_list.append(correctness)

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

async def build_process_supervision_async(inputs, items, num_return_sequences, args):
    """Build process supervision data with async processing"""
    # Initial rollouts - parallelized
    response_list = await build_responses_azure_async(
        inputs,
        num_return_sequences,
        max_new_tokens=args.get('max_new_tokens', 4096),
        temperature=args.get('temperature', 1.0)
    )

    print("Async responses produced:", len(response_list))
    
    # MC scoring - also parallelized
    steps_with_score = await build_mc_scores_async(inputs, response_list, items, num_return_sequences, args)

    outputs = []
    for idx, (response, each_steps_with_score) in enumerate(zip(response_list, steps_with_score)):
        input = inputs[idx // num_return_sequences]
        item = items[idx // num_return_sequences]

        output = item.copy()
        output['response'] = response
        output['steps_with_score'] = each_steps_with_score
        output['question'] = input[0]
        outputs.append(output)

    return outputs

def print_process_supervision(output):
    """Print process supervision output for debugging"""
    steps_with_score = output['steps_with_score']
    print('[Response] Start')
    for step_idx, step in enumerate(steps_with_score):
        print(
            f'[Steps-{step_idx}] Start\n'
            f"{step['step']}\n\n"
            f"Score: {step['score']}\n"
            f"MC Correct: {step['num_mc_correct']}\n"
            f"MC Total: {step['num_mc_total']}\n"
            f'[Steps-{step_idx}] End\n'
        )
    print('[Response] End')

async def main():
    # Configuration parameters
    args = {
        'prompt_path': '/data/users/brandon/ob1-projects/InternVL/internvl_chat/rollout_generation/preprocessed_prompts/preprocessing_scripts/RAVEN/raven_processed_jsonl/center_single_train.jsonl',
        'out_dir': 'raven_rollouts_output_async',
        'batch_size': 8,  # Can increase since we're async
        'num_return_sequences': 4,
        'sample_start_idx': 0,
        'sample_max_num': 50,
        'prompt_version': 'raven_v1',
        'num_mc_sequences': 8,  # Reduced to stay within rate limits
        'max_perception_steps': 12,
        'max_reasoning_steps': 12,
        'early_stop': True,
        'max_new_tokens': 4096,
        'temperature': 1.0,
    }

    # Create output directory
    os.makedirs(args['out_dir'], exist_ok=True)

    print(f"Configuration: {args}")
    print(f"Using Azure OpenAI endpoint: {endpoint}")
    print(f"Model deployment: {deployment}")

    # Load dataset
    dataset = RAVENDataset(
        data=args['prompt_path'],
        sample_max_num=args['sample_max_num'],
        sample_start_idx=args['sample_start_idx'],
    )

    print(f"Dataset loaded: {len(dataset)} samples")

    # Process batches
    batch_size = args['batch_size']
    outputs = []
    sample_times = []
    total_start_time = time.time()

    for i in range(0, min(len(dataset), batch_size)):
        sample = dataset[i]
        inputs = [(sample['rollout_user_prompt'], sample['image'])]
        items = [sample['item']]
        
        print(f"\n[{localtime()}] Processing sample {i+1}/{min(len(dataset), batch_size)}")
        print(f"Image path: {sample['image_path']}")
        print(f"Correct answer: {sample['item']['correct_answer']}")
        
        sample_start_time = time.time()
        
        # Async processing
        curr_outputs = await build_process_supervision_async(
            inputs=inputs,
            items=items,
            num_return_sequences=args['num_return_sequences'],
            args=args
        )
        
        sample_end_time = time.time()
        sample_duration = sample_end_time - sample_start_time
        sample_times.append(sample_duration)
        
        print(f"Sample {i+1} processing time: {sample_duration:.2f} seconds")
        
        outputs.extend(curr_outputs)
        
        if i == 0:
            print("\nFirst sample output:")
            print_process_supervision(curr_outputs[0])

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    print(f"\nGenerated {len(outputs)} rollout samples")

    # Save outputs
    output_file = os.path.join(args['out_dir'], f'raven_rollouts_{args["sample_start_idx"]}_{args["sample_max_num"]}.jsonl')
    with open(output_file, 'w') as f:
        for output in outputs:
            f.write(json.dumps(output) + '\n')

    print(f"Saved {len(outputs)} samples to {output_file}")

    # Print timing statistics
    if sample_times:
        avg_time_per_sample = sum(sample_times) / len(sample_times)
        min_time = min(sample_times)
        max_time = max(sample_times)
        
        print(f"\n=== TIMING STATISTICS ===")
        print(f"Total processing time: {total_duration:.2f} seconds")
        print(f"Average time per sample: {avg_time_per_sample:.2f} seconds")
        print(f"Minimum time per sample: {min_time:.2f} seconds") 
        print(f"Maximum time per sample: {max_time:.2f} seconds")
        
        samples_per_hour = 3600 / avg_time_per_sample if avg_time_per_sample > 0 else 0
        print(f"Estimated throughput: {samples_per_hour:.2f} samples/hour")
        
        if args['sample_max_num']:
            estimated_total_time = avg_time_per_sample * args['sample_max_num']
            estimated_hours = estimated_total_time / 3600
            print(f"Estimated time for {args['sample_max_num']} samples: {estimated_hours:.2f} hours")

if __name__ == "__main__":
    asyncio.run(main()) 