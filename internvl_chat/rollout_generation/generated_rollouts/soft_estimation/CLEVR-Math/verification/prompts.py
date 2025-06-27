verification_prompt = r"""I will provide an abstract visual reasoning problem along with a solution. They will be formatted as follows: 

[Abstract Visual Reasoning Problem]

 <abstract_visual_reasoning_problem>
 ...(abstract visual reasoning problem)... 
</abstract_visual_reasoning_problem> 

[Solution]

<solution>
[Perception]
<step_1>
...(Step 1 of step-by-step perception)...
</step_1>
<step_2>
...(Step 2 of step-by-step perception)...
</step_2>
...
<step_n>
...(Step n of step-by-step perception)...
</step_n>

[Reasoning]
<step_1>
...(Step 1 of step-by-step reasoning)...
</step_1>
<step_2>
...(Step 2 of step-by-step reasoning)...
</step_2>
...
<step_m>
...(Step m of step-by-step reasoning)...
</step_m>

<correct_answer>
...(Clearly state which of the 8 candidate images is the best candidate image as the missing tile to complete the matrix. If the candidates are numbered, lettered, or can be uniquely described, use that identifier.)...
</correct_answer>
</solution>

Your task is to review each paragraph of the solution in sequence, analyzing, verifying, and critiquing the reasoning in detail. You need to provide the analyses and the conclusion in the following format:

```
[Perception]
<analysis_1>
...(analysis of step 1)...
</analysis_1>

...

<analysis_m>
...(analysis of step m)...
</analysis_m>

[Reasoning]
<analysis_1>
...(analysis of step 1)...
</analysis_1>

...

<analysis_n>
...(analysis of step n)...
</analysis_n>

<conclusion>
Correct/Incorrect
</conclusion>
```

* When you analyze each paragraph, you should use proper verification, recalculation, or reflection to indicate whether it is logically and mathematically valid. Please carefully go through this process.

* If an error is detected in any paragraph, you should describe the nature and cause of the error in detail, and suggest how to correct the error or the correct approach. The paragraph is found to contain an error, stop further analysis of subsequent paragraphs (as they may depend on the identified error) and directly provide the conclusion of "Incorrect" in the <conclusion> tag.

For instance, given a solution of five paragraphs, if an error is found in the third paragraph, you should reply in the following format:

<analysis_1>
...(analysis of step 1)...
</analysis_1>

<analysis_2>
...(analysis of step 2)...
</analysis_2>

<analysis_3>
...(analysis of step 3; since an error is found here, also provide detailed critique and correction guideline)...
</analysis_3>

<conclusion>
Incorrect
</conclusion>

Note that the analyses of paragraphs 4 and 5 should be skipped as the paragraph 3 has been found to contain an error.

* Respond with your analyses and conclusion directly in the format above.
------------------------------------------------------------
The following is the abstract visual reasoning problem and the solution for your task:

[Abstract Visual Reasoning Problem]
 <abstract_visual_reasoning_problem>
{{ABSTRACT_VISUAL_REASONING_PROBLEM}}
 </abstract_visual_reasoning_problem>

[Solution]
<solution>
{{SOLUTION}}
</solution>

Remember to:
- Provide only a single string answer of "Correct"/"Incorrect" in the <conclusion> section and no other text or commentary."""