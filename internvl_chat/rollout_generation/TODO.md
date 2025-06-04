# TODO
1. Prepare prompts for RAVEN

1. MC rollouts with Azure OpenAI gpt4.1 for getting soft MC scores first (math-shepherd way of using "future probability" as a score) on small quantity, then scale to full
- Check that postprocessing parsing of steps is correct (include a programmatic check)
2. Collect statisticss to test tool-calling and MSR with o4-mini, gpt4.1
3. Prepare other datasets for rollouts up till 100K