DVQA_V1_ROLLOUT_PROMPT = r"""You are an expert data analyst specializing in interpreting data visualizations. Your task is to answer questions about charts and graphs presented to you in images. You will be provided with an image containing one or more data visualizations and a specific question about the data presented.

I will provide you with an image containing:
- Data Visualization: A chart or graph that contains data points and other visual elements.

Here's the question you need to answer:

<question>
What is the value of the smallest individual bar in the whole chart? Answer the question using a single word or phrase.
</question>

Please follow these steps to complete the task:

1. Carefully examine the image, paying attention to all elements of the data visualization(s) such as titles, labels, axes, legends, and data points.

2. Analyze the data presented in the visualization(s), identifying specific data points or trends that relate to the question asked.

3. Interpret the data and connect it to the specific question asked. Consider how the data directly relates to answering the question.

4. Use your analysis and interpretation to determine the answer to the question. The answer must be a single integer.

5. Present your answer in a LaTeX-formatted box using this format: $\boxed{integer}$

To ensure a thorough and accurate analysis, please structure your response as follows:

[Visual Elements]
Inside your thinking block, list out your step-by-step perception of the visual elements in the chart. Be thorough but concise. Wrap each element in <step> tags and prepend each with a number, counting up.

[Analysis and Interpretation]
Inside your thinking block, explain your step-by-step reasoning process. This should include your analysis, interpretation, and how you arrived at the answer. Provide a clear justification of how you derived the answer from the data presented. Consider multiple possible interpretations before settling on a final answer. Wrap each step in <step> tags.

<correct_answer>
Present your final answer here using the LaTeX-formatted box.
</correct_answer>

It is crucial that your solution contains these sections in the exact format described below:

```
[Visual Elements]
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

[Analysis and Interpretation]
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
$\boxed{integer}$
</correct_answer>
```

Important: Your output must strictly adhere to this format. Do not include any additional text or explanations outside of these sections. Your final output should consist only of the LaTeX-formatted box with the answer and should not duplicate or rehash any of the work you did in the thinking block."""