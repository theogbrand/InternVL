# Verification
1. merge_and_map_batch.py to merge the rollouts into a single jsonl file
    - ```python merge_and_map_batch.py --split="distribute_nine"```
    - model is specified in the JSONL file along with prompt
2. ./run_batch_processor.sh to verify the rollouts, checking for parameters in batch_processor.py