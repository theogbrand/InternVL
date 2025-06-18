1. Transfer all images and their absolute paths to preprocessing_scripts/{dataset_name}
2. Specify the endpoints, deployment and config in rollout.py THEN ./run_rollout.sh to generate rollouts
    - edit check_answer function in rollout.py to match the answer format of the dataset (for RAVEN, option 1-8 ONLY, integer only matching for MMPR correctness prompts, GPT answer checking for open text)
    - check_answer (set prompt_format_version), parse_answer (set scoring_mode)
3. Transfer completed rollouts to generated_rollouts/soft_estimation/{dataset_name}/final_output/{split_name}
4. run ./run_batch_processor.sh to verify the rollouts, checking for parameters in batch_processor.py
    - use test_batch_processor.py to test the batch processor
    - use test_timeout monitor to test recovery for stale Azure Batch Request
    - Use prepare_and_check_batch.ipynb to check status and errors of batches manually based on deployment


Deployment   start_idx   end_idx   row_count
-----------  ----------  --------  ----------
1                 1         682        682
2               683        1364        682
3              1365        2046        682
4              2047        2728        682
5              2729        3410        682
6              3411        4092        682
7              4093        4774        682
8              4775        5456        682
9              5457        6138        682
10             6139        6820        682
11             6821        7502        682
12             7503        8184        682
13             8185        8866        682
14             8867        9548        682
15             9549       10230        682
16            10231       10912        682
17            10913       11594        682
18            11595       12276        682
19            12277       12957        681
20            12958       13638        681
21            13639       14319        681
22            14320       15000        681

AI2D is content filtered error.

InfoVQA, is numbered from 1-9K.