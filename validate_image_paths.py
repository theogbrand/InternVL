import json
import os
from pathlib import Path

def validate_image_paths(jsonl_file):
    """Validate all image paths in the JSONL file and check they correspond to actual JPG files with 'AlgoPuzzleVQA' in filename."""
    
    valid_paths = []
    invalid_paths = []
    missing_files = []
    
    print(f"Validating image paths in: {jsonl_file}")
    print("=" * 60)
    
    # Read the JSONL file
    with open(jsonl_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())
                image_path = record.get('image_path')
                
                if not image_path:
                    print(f"Line {line_num}: Missing 'image_path' field")
                    invalid_paths.append((line_num, None, "Missing image_path field"))
                    continue
                
                # Check if file exists
                if os.path.exists(image_path):
                    # Check if it's a JPG file
                    if image_path.lower().endswith('.jpg'):
                        # Check if it's actually a file (not a directory)
                        if os.path.isfile(image_path):
                            # Check if filename contains "AlgoPuzzleVQA"
                            filename = os.path.basename(image_path)
                            if "AlgoPuzzleVQA" in filename:
                                valid_paths.append((line_num, image_path))
                            else:
                                print(f"Line {line_num}: File exists but filename doesn't contain 'AlgoPuzzleVQA': {image_path}")
                                invalid_paths.append((line_num, image_path, "Missing 'AlgoPuzzleVQA' in filename"))
                        else:
                            print(f"Line {line_num}: Path exists but is not a file: {image_path}")
                            invalid_paths.append((line_num, image_path, "Not a file"))
                    else:
                        print(f"Line {line_num}: File exists but is not JPG: {image_path}")
                        invalid_paths.append((line_num, image_path, "Not JPG file"))
                else:
                    print(f"Line {line_num}: File not found: {image_path}")
                    missing_files.append((line_num, image_path))
                    invalid_paths.append((line_num, image_path, "File not found"))
                    
            except json.JSONDecodeError as e:
                print(f"Line {line_num}: JSON decode error: {e}")
                invalid_paths.append((line_num, None, f"JSON error: {e}"))
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total records processed: {len(valid_paths) + len(invalid_paths)}")
    print(f"Valid JPG files with 'AlgoPuzzleVQA': {len(valid_paths)}")
    print(f"Invalid/missing files: {len(invalid_paths)}")
    print(f"Missing files: {len(missing_files)}")
    
    if valid_paths:
        print(f"\nFirst 5 valid paths:")
        for line_num, path in valid_paths[:5]:
            print(f"  Line {line_num}: {path}")
    
    if missing_files:
        print(f"\nFirst 5 missing files:")
        for line_num, path in missing_files[:5]:
            print(f"  Line {line_num}: {path}")
    
    # Return results for further analysis
    return {
        'valid_paths': valid_paths,
        'invalid_paths': invalid_paths,
        'missing_files': missing_files,
        'total_records': len(valid_paths) + len(invalid_paths)
    }

if __name__ == "__main__":
    # Validate the PuzzleVQA JSONL file
    jsonl_file = "PuzzleVQA_train_run1_1K_v1_subset.jsonl"
    results = validate_image_paths(jsonl_file)
    
    # Additional analysis
    print(f"\nSuccess rate: {len(results['valid_paths'])}/{results['total_records']} = {len(results['valid_paths'])/results['total_records']*100:.1f}%") 