verification_prompt = r"""I will provide a visual reasoning problem along with a solution. They will be formatted as follows, where m and n need not be equal: 

```
[Visual Reasoning Problem]

 <visual_reasoning_problem>
 ...(visual reasoning problem)... 
</visual_reasoning_problem> 

[Solution]

<solution>
[Visual Elements]
<step_1>
...(Step 1 of step-by-step visual elements perception)...
</step_1>
<step_2>
...(Step 2 of step-by-step visual elements perception)...
</step_2>
...
<step_m>
...(Step m of step-by-step visual elements perception)...
</step_m>

[Reasoning]
<step_1>
...(Step 1 of step-by-step reasoning)...
</step_1>
<step_2>
...(Step 2 of step-by-step reasoning)...
</step_2>
...
<step_n>
...(Step n of step-by-step reasoning)...
</step_n>

<correct_answer>
...(The correct answer to the visual reasoning problem)...
</correct_answer>
</solution>
```

Your task is to review each paragraph of the solution in sequence, analyzing, verifying, and critiquing the reasoning in detail. You need to provide the analyses and the conclusion in the following format:

[Visual Elements]
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

* When you analyze each step, you should use proper logical or perceptual verification as appropriate, or reflection to indicate whether it is logically and perceptually valid. Please carefully go through this process.

* Each analysis should:
- Check if the described pattern/rule is actually present
- Verify the logical consistency of the step
- Confirm that conclusions follow from observations

* If an error is detected in any step, you should describe the nature and cause of the error in detail, and suggest how to correct the error or the correct approach. 

* When the step is found to contain an error, stop further analysis of subsequent steps (as they may depend on the identified error) and directly provide the conclusion of "Incorrect" in the <conclusion> tag.

* Only material reasoning or perception errors should trigger the early termination of the analysis.

* For instance, given a solution of five steps, if an error is found in the third step, you should reply in the following format:

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

* Note that the analyses of steps 4 and 5 is skipped as step 3 has been found to contain an error.

------------------------------------------------------------

The following is the visual reasoning problem and its corresponding solution, for you to review and verify:

[Visual Reasoning Problem]
 <visual_reasoning_problem>
{{ABSTRACT_VISUAL_REASONING_PROBLEM}}
 </visual_reasoning_problem>

[Solution]
<solution>
{{SOLUTION}}
</solution>

Remember:
- The <conclusion> tag must contain either Correct or Incorrect, with no additional text or punctuation."""