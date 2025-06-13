1. Transfer all images and their absolute paths to preprocessing_scripts/{dataset_name}
2. Specify the endpoints, deployment and config in rollout.py THEN ./run_rollout.sh to generate rollouts
    - edit check_answer function in rollout.py to match the answer format of the dataset (for RAVEN, option 1-8 ONLY, integer only matching for MMPR correctness prompts, GPT answer checking for open text)
    - check_answer (set prompt_version), parse_answer (set mode)
3. Transfer completed rollouts to generated_rollouts/soft_estimation/{dataset_name}/final_output/{split_name}
4. run ./run_batch_processor.sh to verify the rollouts, checking for parameters in batch_processor.py
    - use test_batch_processor.py to test the batch processor
    - use test_timeout monitor to test recovery for stale Azure Batch Request
    - Use prepare_and_check_batch.ipynb to check status and errors of batches manually based on deployment
