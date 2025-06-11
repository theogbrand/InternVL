import json
import os
import random
import uuid
import tiktoken
import base64
from pathlib import Path
from typing import List, Dict, Any
from mimetypes import guess_type
from prompts import verification_prompt


def merge_jsonl_files(input_folder: str, output_file: str) -> None:
    """Merge all JSONL files in input_folder into a single output file."""
    # Resolve path robustly - handle both absolute and relative paths
    input_path = Path(input_folder).resolve()
    output_path = Path(output_file).resolve()
    
    print(f"Resolved input path: {input_path}")
    print(f"Resolved output path: {output_path}")
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder {input_path} does not exist")
    
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path {input_path} is not a directory")
    
    # Look for both .jsonl and .json files
    jsonl_files = list(input_path.glob("*.jsonl"))
    
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {input_path}")
    
    print(f"Found {len(jsonl_files)} JSONL files to merge")
    
    with open(output_path, 'w') as outfile:
        total_lines = 0
        for jsonl_file in jsonl_files:
            print(f"Processing {jsonl_file.name}...")
            with open(jsonl_file, 'r') as infile:
                file_lines = 0
                for line in infile:
                    line = line.strip()
                    if line:  # Skip empty lines
                        outfile.write(line + '\n')
                        file_lines += 1
                        total_lines += 1
                print(f"  Added {file_lines} lines from {jsonl_file.name}")
    
    print(f"Total merged lines: {total_lines}")
    print(f"Merged file saved as: {output_path}")


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


def extract_content_from_data(data: Dict[str, Any]) -> Dict[str, str]:
    """Extract text content from response key and image path from combined_image_path key."""
    response = data.get('response', '').strip()
    question = data.get('question', '').strip()
    image_path = data.get('combined_image_path', '').strip()
    
    try:
        if not isinstance(response, str) or not response.strip():
            raise ValueError("No text found in key 'response'")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("No text found in key 'question'")
        if not isinstance(image_path, str) or not image_path.strip():
            raise ValueError("No image path found in key 'combined_image_path'")
    except ValueError as e:
        raise e
    
    return {
        'response': response,
        'question': question,
        'image_path': image_path
    }


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in text using tiktoken."""
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception as e:
        print(f"Error counting tokens: {e}")
        # Fallback to rough word-based estimation
        return len(text.split()) * 1.3  # Rough approximation


def calculate_average_tokens(jsonl_file: str, sample_size: int = 1000) -> float:
    """Calculate average tokens per JSONL object from a random sample."""
    
    # Resolve path robustly
    file_path = Path(jsonl_file).resolve()
    print(f"Processing file: {file_path}")
    
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist")
    
    # First, count total lines
    with open(file_path, 'r') as f:
        all_lines = [line.strip() for line in f if line.strip()]
    
    total_lines = len(all_lines)
    print(f"Total lines in file: {total_lines}")
    
    # Sample lines
    if total_lines <= sample_size:
        sample_lines = all_lines
        print(f"Using all {total_lines} lines (less than requested sample size)")
    else:
        sample_lines = random.sample(all_lines, sample_size)
        print(f"Randomly sampled {sample_size} lines")
    
    token_counts = []
    
    for i, line in enumerate(sample_lines):
        try:
            data = json.loads(line)
            
            # Extract text content and image path
            content = extract_content_from_data(data)
            question = content['question']
            solution = content['response']
            image_path = content['image_path']
            
            total_content = question + solution
            
            # Convert image to base64 and add to content for token calculation
            if image_path and os.path.exists(image_path):
                try:
                    base64_image = local_image_to_data_url(image_path)
                    total_content += f" {base64_image}"
                    print(f"Line {i+1}: Converted image {image_path} to base64 for token calculation")
                except Exception as e:
                    print(f"Warning: Failed to convert image {image_path} to base64: {e}")
            elif image_path:
                print(f"Warning: Image path {image_path} does not exist for line {i+1}")
            
            if total_content:
                token_count = count_tokens(total_content)
                token_counts.append(token_count)
            else:
                print(f"Warning: No content found in line {i+1}")
                
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON in line {i+1}: {e}")
            continue
        except Exception as e:
            print(f"Error processing line {i+1}: {e}")
            continue
    
    if not token_counts:
        raise ValueError("No valid token counts calculated")
    
    average_tokens = sum(token_counts) / len(token_counts)
    
    if average_tokens <= 0:
        raise ValueError(f"Average tokens must be positive, got: {average_tokens}")
    
    print(f"\nToken count statistics:")
    print(f"Valid samples processed: {len(token_counts)}")
    print(f"Min tokens: {min(token_counts)}")
    print(f"Max tokens: {max(token_counts)}")
    print(f"Average tokens: {average_tokens:.2f}")
    
    return average_tokens


def count_total_lines(jsonl_file: str) -> int:
    """Count total lines in the merged JSONL file."""
    file_path = Path(jsonl_file).resolve()
    
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist")
    
    with open(file_path, 'r') as f:
        total_lines = sum(1 for line in f if line.strip())
    
    print(f"Total lines in merged file: {total_lines:,}")
    return total_lines


def calculate_batch_requirements(total_lines: int, avg_tokens_per_line: float, max_tokens_per_batch: int = 1_000_000_000) -> Dict[str, Any]:
    """Calculate batch requirements given total lines and token constraints."""
    
    print(f"\nCalculating batch requirements:")
    print(f"Total lines: {total_lines:,}")
    print(f"Average tokens per line: {avg_tokens_per_line:.2f}")
    print(f"Max tokens per batch: {max_tokens_per_batch:,}")
    
    # Check for zero division
    if avg_tokens_per_line <= 0:
        raise ValueError(f"Average tokens per line must be positive, got: {avg_tokens_per_line}")
    
    if total_lines <= 0:
        raise ValueError(f"Total lines must be positive, got: {total_lines}")
    
    # Calculate lines per batch using 95% of max tokens to maximize throughput while staying safe
    effective_max_tokens = max_tokens_per_batch * 0.95
    lines_per_batch = int(effective_max_tokens // avg_tokens_per_line)
    print(f"Using 95% of max tokens ({effective_max_tokens:,.0f}) to maximize throughput")
    
    # Ensure at least 1 line per batch
    if lines_per_batch <= 0:
        lines_per_batch = 1
        print(f"Warning: Very large tokens per line ({avg_tokens_per_line:.0f}), setting lines_per_batch to 1")
    
    # Calculate number of batches needed
    total_batches = (total_lines + lines_per_batch - 1) // lines_per_batch  # Ceiling division
    
    # Calculate actual tokens per batch (except possibly the last batch)
    tokens_per_batch = lines_per_batch * avg_tokens_per_line
    
    # Calculate lines in the last batch
    lines_in_last_batch = total_lines % lines_per_batch
    if lines_in_last_batch == 0:
        lines_in_last_batch = lines_per_batch
    
    tokens_in_last_batch = lines_in_last_batch * avg_tokens_per_line
    
    print(f"\nBatch Requirements:")
    print(f"Lines per batch: {lines_per_batch:,}")
    print(f"Total batches needed: {total_batches}")
    print(f"Tokens per batch: {tokens_per_batch:,.0f} ({tokens_per_batch/max_tokens_per_batch*100:.1f}% of limit, {tokens_per_batch/effective_max_tokens*100:.1f}% of target)")
    print(f"Lines in last batch: {lines_in_last_batch:,}")
    print(f"Tokens in last batch: {tokens_in_last_batch:,.0f} ({tokens_in_last_batch/max_tokens_per_batch*100:.1f}% of limit, {tokens_in_last_batch/effective_max_tokens*100:.1f}% of target)")
    
    return {
        "total_lines": total_lines,
        "avg_tokens_per_line": avg_tokens_per_line,
        "max_tokens_per_batch": max_tokens_per_batch,
        "effective_max_tokens": effective_max_tokens,
        "lines_per_batch": lines_per_batch,
        "total_batches": total_batches,
        "tokens_per_batch": tokens_per_batch,
        "lines_in_last_batch": lines_in_last_batch,
        "tokens_in_last_batch": tokens_in_last_batch
    }


def split_jsonl_into_batches(merged_file: str, batch_output_dir: str, lines_per_batch: int, total_batches: int) -> List[str]:
    """Split merged JSONL file into separate batch files, transforming to OpenAI batch API format."""
    
    # Resolve paths
    merged_path = Path(merged_file).resolve()
    batch_dir = Path(batch_output_dir).resolve()
    
    print(f"\nSplitting JSONL file into batches:")
    print(f"Source file: {merged_path}")
    print(f"Batch directory: {batch_dir}")
    print(f"Lines per batch: {lines_per_batch:,}")
    print(f"Total batches: {total_batches}")
    
    if not merged_path.exists():
        raise FileNotFoundError(f"Merged file {merged_path} does not exist")
    
    # Create batch output directory
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    batch_files = []
    current_batch = 1
    lines_in_current_batch = 0
    current_batch_file = None
    current_batch_path = None
    
    try:
        with open(merged_path, 'r') as infile:
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                # Parse the input JSON
                try:
                    input_data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON in line {line_num}: {e}")
                    continue
                
                # Extract required data
                try:
                    content = extract_content_from_data(input_data)
                    question = content['question']
                    solution = content['response']
                    image_path = content['image_path']
                    
                    # Get custom_id from original data structure
                    custom_id = input_data.get('id', f'request-{line_num}')
                    if 'subset_split' in input_data:
                        custom_id = f"{custom_id}_{input_data['subset_split']}_{uuid.uuid4()}"
                    
                    # Convert image to base64
                    if not image_path or not os.path.exists(image_path):
                        print(f"Warning: Image path {image_path} does not exist for line {line_num}, skipping")
                        continue
                        
                    base64_image_url = local_image_to_data_url(image_path)
                    
                    # Transform to OpenAI batch API format
                    transformed_data = {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/chat/completions",
                        "body": {
                            "model": "o4-mini",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": verification_prompt.replace("{{ABSTRACT_VISUAL_REASONING_PROBLEM}}", question).replace("{{SOLUTION}}", solution)
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": base64_image_url
                                            }
                                        }
                                    ]
                                }
                            ],
                            "max_completion_tokens": 4096
                        }
                    }
                    
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    continue
                
                # Start new batch if needed
                if lines_in_current_batch == 0:
                    if current_batch_file:
                        current_batch_file.close()
                    
                    batch_filename = f"batch_{current_batch:04d}.jsonl"
                    current_batch_path = batch_dir / batch_filename
                    current_batch_file = open(current_batch_path, 'w')
                    batch_files.append(str(current_batch_path))
                    print(f"Creating batch {current_batch}: {batch_filename}")
                
                # Write transformed line to current batch
                current_batch_file.write(json.dumps(transformed_data) + '\n')
                lines_in_current_batch += 1
                
                # Check if current batch is full
                if lines_in_current_batch >= lines_per_batch:
                    current_batch_file.close()
                    print(f"  Completed batch {current_batch}: {lines_in_current_batch:,} lines")
                    current_batch += 1
                    lines_in_current_batch = 0
                    current_batch_file = None
        
        # Close the last batch file if it's still open
        if current_batch_file:
            current_batch_file.close()
            print(f"  Completed batch {current_batch-1}: {lines_in_current_batch:,} lines")
    
    except Exception as e:
        # Ensure we close any open file handles on error
        if current_batch_file:
            current_batch_file.close()
        raise e
    
    print(f"\nSuccessfully created {len(batch_files)} batch files:")
    for i, batch_file in enumerate(batch_files, 1):
        file_path = Path(batch_file)
        with open(file_path, 'r') as f:
            line_count = sum(1 for line in f if line.strip())
        print(f"  Batch {i}: {file_path.name} ({line_count:,} lines)")
    
    return batch_files


def main():
    # Configuration
    input_folder = "/data/users/brandon/ob1-projects/InternVL/internvl_chat/rollout_generation/generated_rollouts/soft_estimation/RAVEN/final_output"
    output_dir = "/data/users/brandon/ob1-projects/InternVL/internvl_chat/rollout_generation/generated_rollouts/soft_estimation/RAVEN/verification/verification_pipeline_outputs"
    merged_file = os.path.join(output_dir, "merged_batch_output.jsonl")
    batch_output_dir = os.path.join(output_dir, "verification_batches")
    sample_size = 1000
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    print("=== JSONL Merger and Token Calculator ===\n")
    
    # Step 1: Merge JSONL files
    print("Step 1: Merging JSONL files...")
    try:
        merge_jsonl_files(input_folder, merged_file)
    except Exception as e:
        print(f"Error during merging: {e}")
        return
    
    print("\n" + "="*50 + "\n")
    
    # Step 2: Calculate average tokens
    print("Step 2: Calculating average tokens...")
    try:
        avg_tokens = calculate_average_tokens(merged_file, sample_size)
        print(f"\n🎯 Average tokens per JSONL object = {avg_tokens:.2f}")
    except Exception as e:
        print(f"Error during token calculation: {e}")
        return
    
    print("\n" + "="*50 + "\n")
    
    # Step 3: Calculate batch requirements from merged file
    print("Step 3: Calculating batch requirements (1B token limit)...")
    try:
        total_lines = count_total_lines(merged_file)
        batch_requirements = calculate_batch_requirements(total_lines, avg_tokens, max_tokens_per_batch=10_000_000)
        
        # Save batch requirements to file
        batch_info_file = os.path.join(output_dir, "batch_requirements.json")
        
        with open(batch_info_file, 'w') as f:
            json.dump(batch_requirements, f, indent=2)
        
        print(f"\n🎯 Batch requirements saved to: {batch_info_file}")
        print(f"🎯 Total batches needed: {batch_requirements['total_batches']}")
        print(f"🎯 Lines per batch: {batch_requirements['lines_per_batch']:,}")
        
    except Exception as e:
        print(f"Error during batch calculation: {e}")
        return
    
    print("\n" + "="*50 + "\n")
    
    # Step 4: Split merged file into batch files
    print("Step 4: Splitting merged file into batch files...")
    try:
        batch_files = split_jsonl_into_batches(
            merged_file=merged_file,
            batch_output_dir=batch_output_dir,
            lines_per_batch=batch_requirements['lines_per_batch'],
            total_batches=batch_requirements['total_batches']
        )
        
        # Save batch file list
        batch_files_info = {
            "batch_output_directory": batch_output_dir,
            "total_batch_files": len(batch_files),
            "batch_files": batch_files,
            "lines_per_batch": batch_requirements['lines_per_batch']
        }
        
        batch_files_info_file = os.path.join(output_dir, "batch_files_info.json")
        with open(batch_files_info_file, 'w') as f:
            json.dump(batch_files_info, f, indent=2)
        
        print(f"\n🎯 Successfully created {len(batch_files)} batch files in: {batch_output_dir}")
        print(f"🎯 Batch files info saved to: {batch_files_info_file}")
        
    except Exception as e:
        print(f"Error during batch file creation: {e}")
        return


if __name__ == "__main__":
    main()
