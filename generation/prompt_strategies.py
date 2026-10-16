import re
from enum import Enum


class PromptStyle(Enum):
    Baseline = 0      # original ClassEval prompt: instruction + skeleton
    CoT = 1           # chain-of-thought: reason about fields/dependencies first, then code
    FewShot = 2       # k solved example classes from ClassEval, then the target
    PlanThenCode = 3  # two calls: 1) write an implementation plan, 2) code following the plan


DEFAULT_FEWSHOT_IDS = ["ClassEval_6", "ClassEval_30"]

FINAL_CODE_RULE = ("Output the complete final class (including imports) in exactly one ```python code block "
                   "at the end of your answer. Do not change the class name or method signatures.")

COT_TEMPLATE = """Please complete the class {class_name} in the following code.
{skeleton}

Before writing code, think step by step:
1. List the fields initialized in __init__ and what each one stores.
2. For every method, describe its behaviour, the edge cases in its docstring examples, and which fields or other methods of the class it reads, modifies, or calls.
3. Point out dependencies between methods and decide the order in which they should be implemented.
Then write the implementation.
{final_rule}"""

FEWSHOT_EXAMPLE_TEMPLATE = """### Example {idx}
Skeleton:
```python
{skeleton}
```
Solution:
```python
{solution}
```
"""

FEWSHOT_TEMPLATE = """You will complete a Python class from its skeleton. Here are solved examples.

{examples}
### Task
Please complete the class {class_name} in the following code.
{skeleton}

{final_rule}"""

PLAN_TEMPLATE = """Below is a class skeleton for {class_name}.
{skeleton}

Do NOT write code yet. Write a concise implementation plan:
- the role of every field in __init__,
- for each method: its steps, edge cases, and which fields/methods of the class it depends on,
- any shared helper logic or library imports needed."""

CODE_FROM_PLAN_TEMPLATE = """Please complete the class {class_name} in the following code.
{skeleton}

Follow this implementation plan:
{plan}

{final_rule}"""


def build_fewshot_examples(data, fewshot_ids):
    by_id = {cont['task_id']: cont for cont in data}
    examples = []
    for i, task_id in enumerate(fewshot_ids, 1):
        cont = by_id[task_id]
        examples.append(FEWSHOT_EXAMPLE_TEMPLATE.format(idx=i, skeleton=cont['skeleton'].strip(),
                                                        solution=cont['solution_code'].strip()))
    return '\n'.join(examples)


def cot_instruction(cont):
    return COT_TEMPLATE.format(class_name=cont['class_name'], skeleton=cont['skeleton'], final_rule=FINAL_CODE_RULE)


def fewshot_instruction(cont, examples):
    return FEWSHOT_TEMPLATE.format(examples=examples, class_name=cont['class_name'], skeleton=cont['skeleton'],
                                   final_rule=FINAL_CODE_RULE)


def plan_instruction(cont):
    return PLAN_TEMPLATE.format(class_name=cont['class_name'], skeleton=cont['skeleton'])


def code_from_plan_instruction(cont, plan):
    return CODE_FROM_PLAN_TEMPLATE.format(class_name=cont['class_name'], skeleton=cont['skeleton'], plan=plan.strip(),
                                          final_rule=FINAL_CODE_RULE)


def santacoder_cot_instruction(cont):
    return ("# Complete the Python class below. First consider the fields initialized in __init__, "
            "each method's behavior and edge cases, and dependencies between methods. Then continue "
            "the code directly; do not repeat the class or add prose.\n\n" + cont['skeleton'].rstrip())


def santacoder_fewshot_instruction(cont, examples):
    return ("# Learn the class implementation pattern from these solved examples.\n\n" + examples.strip() +
            "\n\n# Complete the target class by continuing its code. Do not repeat the class or add prose.\n\n" +
            cont['skeleton'].rstrip())


def santacoder_plan_instruction(cont):
    return ("# Analyze this class skeleton and write a concise implementation plan only. Include the role "
            "of each field, each method's behavior and edge cases, and method dependencies. Do not write code.\n\n" +
            cont['skeleton'].rstrip() + "\n\n# Implementation plan:\n")


def santacoder_code_from_plan_instruction(cont, plan):
    return ("# Implement the target class according to this plan. Continue the Python code directly; "
            "do not repeat the class or add prose.\n# Plan:\n" + plan.strip() + "\n\n" + cont['skeleton'].rstrip())


def strip_response(output):
    # local HF models return prompt + completion; keep only the completion
    for identifier in ["### Response:", "@@ Response:", "[/INST]"]:
        if identifier in output:
            return output.split(identifier)[-1]
    return output


def extract_final_code(output, class_name=None):
    # the evaluator takes the FIRST ```python block, but CoT/plan answers may contain snippets
    # during reasoning, so keep only the LAST block that defines a class
    output = strip_response(output)
    blocks = re.findall(r"```(?:python)?\n(.*?)```", output, re.S)
    class_blocks = [b for b in blocks if re.search(r"^class\s", b, re.M)]
    if class_blocks:
        return "```python\n" + class_blocks[-1] + "```"
    if class_name:
        matches = list(re.finditer(rf"^class\s+{re.escape(class_name)}\b", output, re.M))
        if matches:
            return output[matches[-1].start():]
    return output
