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

If content filter error thrown, API will return response "Content filter error", which results in a "fail to parse" error, and the prompt will be skipped, with error "fail to parse" in the log.

Example: infovqa_run1_open_ans_9K_v1_subset_raven_rollouts_1289_1610_streaming has 8 "failed to parse" errors, which are all content filter errors, as can see in log, "failed to parse: 4/60 rollouts" twice. 

InfoVQA, is numbered from 1-9K.

Endpoint 1:  IDs 1-322 (322 samples)
Endpoint 2:  IDs 323-644 (322 samples)
Endpoint 3:  IDs 645-966 (322 samples)
Endpoint 4:  IDs 967-1288 (322 samples) done
Endpoint 5:  IDs 1289-1610 (322 samples) done
Endpoint 6:  IDs 1611-1932 (322 samples)
Endpoint 7:  IDs 1933-2254 (322 samples)
Endpoint 8:  IDs 2255-2576 (322 samples)
Endpoint 9:  IDs 2577-2898 (322 samples)
Endpoint 10: IDs 2899-3220 (322 samples)
Endpoint 11: IDs 3221-3542 (322 samples)
Endpoint 12: IDs 3543-3864 (322 samples)
Endpoint 13: IDs 3865-4185 (321 samples)
Endpoint 14: IDs 4186-4506 (321 samples)
Endpoint 15: IDs 4507-4827 (321 samples)
Endpoint 16: IDs 4828-5148 (321 samples)
Endpoint 17: IDs 5149-5469 (321 samples)
Endpoint 18: IDs 5470-5790 (321 samples)
Endpoint 19: IDs 5791-6111 (321 samples)
Endpoint 20: IDs 6112-6432 (321 samples)
Endpoint 21: IDs 6433-6753 (321 samples)
Endpoint 22: IDs 6754-7074 (321 samples)
Endpoint 23: IDs 7075-7395 (321 samples)
Endpoint 24: IDs 7396-7716 (321 samples)
Endpoint 25: IDs 7717-8037 (321 samples)
Endpoint 26: IDs 8038-8358 (321 samples)
Endpoint 27: IDs 8359-8679 (321 samples)
Endpoint 28: IDs 8680-9000 (321 samples)