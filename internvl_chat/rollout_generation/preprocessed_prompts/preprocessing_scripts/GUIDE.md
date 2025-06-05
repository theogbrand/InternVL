In general, 

1. Export all images (with absolute paths) and their corresponding prompts into the same directory (to prevent incorrect path issues when loading image for inference)
2. Flatten all prompts (jsonl item) into a single jsonl file
3. Run small batch inference using this flattened jsonl file containing all prompts and test if image file is loaded correctly