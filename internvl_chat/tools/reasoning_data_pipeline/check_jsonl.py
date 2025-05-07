import json
from collections import defaultdict
import os
import argparse

def find_incomplete_ids(filepath: str, max_id: int, expected_count: int = 4) -> list[int]:
    """
    Loads a JSONL file, counts occurrences of integer IDs (assumed to be in
    the 'image' field) within the range [1, max_id], and returns IDs that
    are missing or do not have the expected count.

    Args:
        filepath: Path to the JSONL file.
        max_id: The maximum integer ID to check for (inclusive).
        expected_count: The expected number of generations per ID.

    Returns:
        A list of integer IDs from 1 to max_id that are missing or have an
        incorrect count. Returns an empty list if the file doesn't exist.
        Prints error/warning messages for file issues or parsing problems.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return []

    id_counts = defaultdict(int)
    processed_ids = set()
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    if 'id' in item:
                        try:
                            # Assume 'image' holds the integer ID
                            img_id = int(item['id'])
                            if 1 <= img_id <= max_id:
                                id_counts[img_id] += 1
                                processed_ids.add(img_id)
                            # Silently ignore IDs outside the target range
                        except (ValueError, TypeError):
                             print(f"Warning: Skipping line {line_num} due to non-integer 'id' field: {item.get('id')}")
                    else:
                        print(f"Warning: Skipping line {line_num} due to missing 'id' field: {line.strip()}")
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line {line_num}: {line.strip()}")
    except IOError as e:
        print(f"Error opening or reading file {filepath}: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []

    incomplete_ids = []
    # Check all IDs from 1 to max_id
    for i in range(1, max_id + 1):
        count = id_counts.get(i, 0) # Get count, default to 0 if missing
        if count != expected_count:
            incomplete_ids.append(i)

    if not processed_ids and max_id > 0 :
         print(f"Warning: No valid data within the range [1, {max_id}] found or processed in {filepath}")
         # Still return the full list of missing IDs in this case
         return list(range(1, max_id + 1))


    return incomplete_ids

# Example usage:
# python /home/ncs/ob1/InternVL/internvl_chat/tools/reasoning_data_pipeline/check_jsonl.py "/home/ncs/ob1/InternVL/internvl_chat/sampled_outputs/OpenGVLab_InternVL3-8B/max_tiles_6/MARVEL_AVR_flattened_10.jsonl" 10 --expected_count 4
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check JSONL file for incomplete IDs.")
    # Positional arguments
    parser.add_argument("file_path", help="Path to the JSONL file.")
    parser.add_argument("max_target_id", type=int, help="Maximum integer ID to check (inclusive).")
    # Optional argument
    parser.add_argument(
        "--expected_count",
        type=int,
        default=4,
        help="Expected number of generations per ID (default: 4)."
    )

    args = parser.parse_args()

    # Use arguments from command line
    # Note: variable names match the function parameters now
    missing_or_incorrect_ids = find_incomplete_ids(args.file_path, args.max_target_id, args.expected_count)

    if missing_or_incorrect_ids:
        print(f"Found {len(missing_or_incorrect_ids)} IDs in range [1, {args.max_target_id}] missing or with incorrect count ({args.expected_count}):")
        # Print first 10 for brevity if many are missing/incorrect
        print(f"  {missing_or_incorrect_ids[:10]}{'...' if len(missing_or_incorrect_ids) > 10 else ''}")

    else:
        # Check if the function returned an empty list because the file didn't exist
        # versus all IDs being present with the correct count.
        # Use args.file_path here as well
        if os.path.exists(args.file_path):
             print(f"All IDs from 1 to {args.max_target_id} have the expected count ({args.expected_count}).")
        # else: file not found case handled by the function itself