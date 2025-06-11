import json
import os
import random
import tiktoken
import base64
from pathlib import Path
from typing import List, Dict, Any
from mimetypes import guess_type


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
    response = data.get('response', '')
    image_path = data.get('combined_image_path', '')
    
    try:
        if not isinstance(response, str) or not response.strip():
            raise ValueError("No text found in key 'response'")
        text_content = response
    except ValueError as e:
        raise e
    
    return {
        'text': text_content,
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
            text_content = content['text']
            image_path = content['image_path']
            
            total_content = text_content
            
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
    
    # Calculate lines per batch using 90% of max tokens to maximize throughput while staying safe
    effective_max_tokens = max_tokens_per_batch * 0.9
    lines_per_batch = int(effective_max_tokens // avg_tokens_per_line)
    print(f"Using 90% of max tokens ({effective_max_tokens:,.0f}) to maximize throughput")
    
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


def main():
    # Configuration
    input_folder = "/data/users/brandon/ob1-projects/InternVL/internvl_chat/rollout_generation/generated_rollouts/soft_estimation/RAVEN/final_output"
    output_dir = "/data/users/brandon/ob1-projects/InternVL/internvl_chat/rollout_generation/generated_rollouts/soft_estimation/RAVEN/verification/verification_pipeline_outputs"
    merged_file = os.path.join(output_dir, "merged_batch_output.jsonl")
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
        batch_requirements = calculate_batch_requirements(total_lines, avg_tokens, max_tokens_per_batch=300_000)
        
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


if __name__ == "__main__":
    main()
