import json
import os
import glob
import multiprocessing
from typing import List, Tuple

def convert_label_data(input_path: str, output_path: str) -> None:
    """
    Converts label data from the input JSON format to the target format.
    (Handles a single file)

    Args:
        input_path: Path to the input JSON file (e.g., 'Json_data/1/1_label.json').
        output_path: Path to save the formatted JSON data.
    """
    try:
        with open(input_path, 'r') as f_in:
            data = json.load(f_in)

        image_dir = os.path.dirname(input_path)
        image_filename = f"{data['id']}.png" # Assuming PNG format
        image_path = os.path.join(image_dir, image_filename)

        formatted_data = [
            {
                "id": str(data["id"]),
                "question": data["avr_question"],
                "correct_answer": str(data["answer"]),
                "image_path": image_path
            }
        ]
        # Ensure the output directory exists before writing
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f_out:
            json.dump(formatted_data, f_out, indent=4)

        # Optional: Print success per file (can be noisy in parallel)
        # print(f"Successfully formatted '{input_path}' to '{output_path}'")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_path}'")
    except KeyError as e:
        print(f"Error: Missing key {e} in input file '{input_path}'")
    except Exception as e:
        print(f"Error processing '{input_path}': {e}")

def generate_task_args(input_dir_pattern: str, output_base_dir: str) -> List[Tuple[str, str]]:
    """
    Generates a list of (input_path, output_path) tuples for processing.

    Args:
        input_dir_pattern: Glob pattern to find input JSON files (e.g., 'Json_data/*/*_label.json').
        output_base_dir: The base directory where formatted files will be saved.

    Returns:
        A list of tuples, where each tuple is (input_file_path, output_file_path).
    """
    task_args = []
    input_files = glob.glob(input_dir_pattern, recursive=True) # Use recursive=True if needed for nested dirs

    if not input_files:
        print(f"Warning: No input files found matching pattern: {input_dir_pattern}")
        return []

    for input_file in input_files:
        output_file_name = os.path.basename(input_file).replace('_label.json', '_formatted.json')
        output_file = os.path.join(output_base_dir, output_file_name)
        task_args.append((input_file, output_file))

    return task_args

def run_parallel_conversion(input_pattern: str, output_dir: str, num_processes: int = None):
    """
    Runs the data conversion in parallel using multiprocessing.

    Args:
        input_pattern: Glob pattern for input files.
        output_dir: Base directory for output files.
        num_processes: Number of parallel processes to use. Defaults to os.cpu_count().
    """
    task_args = generate_task_args(input_pattern, output_dir)
    if not task_args:
        print("No tasks to process.")
        return

    # Sort task_args numerically based on the input filename ID
    def get_sort_key(task_tuple):
        input_path = task_tuple[0]
        basename = os.path.basename(input_path)
        # Assumes filename format like '123_label.json'
        try:
            return int(basename.split('_')[0])
        except (ValueError, IndexError):
            # Handle cases where filename doesn't match expected format
            print(f"Warning: Could not parse ID from filename: {basename}. Placing it at the end.")
            return float('inf') # Place unparsable names at the end

    task_args.sort(key=get_sort_key)

    # Limit to the first 50 tasks for testing
    if len(task_args) > 50:
        print(f"Limiting processing to the first 50 out of {len(task_args)} files for testing.")
        task_args = task_args[:50]

    print(f"Found {len(task_args)} files to process. Starting parallel conversion...")

    # Use a context manager for the pool to ensure proper cleanup
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use starmap to pass multiple arguments (input_path, output_path)
        # from each tuple in task_args to convert_label_data
        results = pool.starmap(convert_label_data, task_args)
        # Note: convert_label_data returns None, so results will be a list of None
        # If the function returned values, they'd be collected here.
        # Errors raised within convert_label_data in worker processes will be
        # raised here when results are accessed or the pool closes.

    print(f"Parallel conversion finished for {len(task_args)} files.")

# --- Example Usage ---
if __name__ == "__main__":
    # Important: Protect multiprocessing code execution within this block

    # Define where to find input files and where to save output
    # This pattern assumes folders like Json_data/1/, Json_data/2/, etc.
    # Adjust the pattern if your directory structure is different
    input_glob_pattern = '/mnt/weka/aisg/ob1/raw_datasets/MARVEL_AVR/Json_data/*/*_label.json'
    output_directory = '/mnt/weka/aisg/ob1/MM-PRM/formatted_data/MARVEL_AVR' # A separate directory for output

    # Create the main output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Run the conversion, leave num_processes=None to use all available CPU cores
    run_parallel_conversion(input_glob_pattern, output_directory)

    # Example: Verify one of the outputs if needed
    # try:
    #     verify_file = os.path.join(output_directory, '1', '1_formatted.json')
    #     with open(verify_file, 'r') as f:
    #         print(f"\n--- Content of {verify_file} ---")
    #         print(f.read())
    # except FileNotFoundError:
    #     print(f"Verification file {verify_file} not found (conversion might have failed or pattern mismatch).")