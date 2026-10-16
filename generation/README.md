## Generation 
We provide `inference.py` script to to produce class-level code outputs, supporting various models and generation strategies.

### Local demo UI

From the repository root, run `venv\Scripts\python.exe local_demo.py`, then open the displayed localhost URL if a browser does not open automatically. Choose a ClassEval task, local model, and prompt style, then click **Chạy model → JSON**. The UI saves one-task results under `output/model_output`, shows the generated class, and lets you inspect or download the saved JSON. You can also open an existing output JSON without generating again. The first use of a Hugging Face model ID downloads its checkpoint into `model_cache/huggingface` inside this project; a local directory path uses files already on disk. The UI includes SantaCoder and DeepSeek-Coder-Instruct 1.3B with CoT, Few-shot, and Plan-then-code, plus the paper's local model families and CodeGeeX2. The UI runs one selected task at a time.

The **So sánh chỉ số** page reads the single merged source `output/result/pass_at_k_result.json`. It shows class and method success rates for greedy Pass@1 and sampled Pass@1/3/5, with filters for model and strategy. Gemini CoT, few-shot, and plan results from ClassEval2 are included. Generating a single task does not automatically add a benchmark score.

### API credentials

Create a private `.env` file in the repository root using `.env.example` as a template. Set `GOOGLE_API_KEY` for Gemini or `OPENAI_API_KEY` for GPT models. `OPENAI_BASE_URL` is optional. The generation script loads `.env` automatically; a value supplied through the command line takes precedence. Local Transformers models do not need either API key. Never put real keys in the README or command history.

For example, from `generation`:

```bash
python inference.py --model 14 --prompt_style 1 --max_length 8192 --output_path gemini_cot.json
```

To try the paper's smallest local model with the same ClassEval JSON output format:

```bash
python inference.py --model 5 --checkpoint bigcode/santacoder --data_path ../data/ClassEval_data.json --output_path ../output/model_output/santacoder_greedy.json --greedy 1 --generation_strategy 0
```

On Windows with an NVIDIA GPU, install the CUDA-enabled dependencies from the repository root with `venv\Scripts\python.exe -m pip install -r requirements-gpu.txt`. This pins PyTorch 2.5.1 with CUDA 12.4 and NumPy 1.23.5, which also avoids the NumPy ABI warning from the previous CPU environment. Confirm GPU access with `venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`. Local models use CUDA when available and otherwise run on CPU. The first run downloads the chosen checkpoint. To score the generated file, run `python evaluation.py --source_file_name santacoder_greedy --greedy 1` from `classeval_evaluation`.

Navigate to the generation directory and execute the following command:
```
python inference.py \
--model 0 \
--checkpoint model_path \
--output_path model_output.json \
--greedy 1 \
--data_path ../data/ClassEval_data.json \
--cuda 0 1 \
--generation_strategy 0

```

## Parameter Guide

### Supported Models

Specify the model using the `--model` argument. Each model might have its own unique API.

Supported models:

0: Instruct_CodeGen
1: WizardCoder
2: Instruct_StarCoder
...

For a detailed list of models and their respective identifiers, check the help associated with the `--model` parameter in `inference.py`. To integrate new models, append the appropriate model identifier in `inference.py` and ensure its API is also added.

Currently, we cater to 9 models: Instruct_CodeGen, WizardCoder, Instruct_StarCoder, InCoder, PolyCoder, SantaCoder, Vicuna, ChatGLM, GPT-3.5, and GPT-4. For GPT-3.5 and GPT-4, supply your personal OpenAI key using the `--openai_key` parameter.

### Sampling Methods

Specify your desired sampling method with the `--greedy` argument:

- 0: Nucleus sampling. 
- 1: Greedy sampling.

For nucleus sampling, ensure you also provide the sample number using the `--sample` argument.

### Generation Strategies
Determine your generation strategy with the `--generation_strategy` argument:

- 0: Holistic Generation
- 1: Incremental Generation
- 2: Compositional Generation

For insights on the practical implementation of these generation strategies, see the `def pipeline` within `inference_pipeline.py`.

### Prompt Styles
With Holistic Generation (`--generation_strategy 0`), choose a prompt style with `--prompt_style` (templates in `prompt_strategies.py`):

- 0: Baseline (original ClassEval prompt)
- 1: Chain-of-Thought: reason about fields, method behaviour and inter-method dependencies before coding
- 2: Few-shot: prepend solved ClassEval classes given by `--fewshot_ids` (default `ClassEval_6 ClassEval_30`); these tasks are excluded from generation
- 3: Plan-then-Code: one call writes an implementation plan, a second call writes the class following it

For styles 1-3, `predict` keeps only the last ```python block that defines a class, and the full model answer is saved in `raw_output`.

### Further Parameters

For a comprehensive overview of all parameters and their valid values, either peruse the `inference.py` script or run with the `--help` flag.

### Prompt Desgin


We can categorize all studied LLMs into three distinct classes based on their capabilities: LLMs with Instruction-Following (IF) ability via instruction tuning, LLMs without instruction tuning but possessing the ability of "filling-in-the-middle" (FIM), and LLMs lacking both IF and FIM abilities.

#### LLMs with IF Ability

Following the common practice of prompting LLMs with IF ability like WizardCoder, we set their prompts
of two parts: (i) a system prompt as the beginning sentence to initialize the model, and followed by (ii) a task instruction to describe the goal of the task.

**System Prompt**: *Provided below is an instruction detailing a task. Compose a response that aptly fulfills the request.*

**Instruction-H**: *Please complete the class **${Class Name}** in the subsequent code. **${Class Skeleton}***

Refer to Figure 2 in the paper for details on the class skeleton.

**Instruction-I**: *Please complete the method **${Method Name}** within the following class **${Class Name}. ${Class-level Info} ${Generated Methods with Contract Designs} ${Target Method Contract Design}***

We provide a more explicit representation of the incremental generation process in the following figure: 

<img src="incremental generation.png" alt="Incremental Generation Process" style="zoom: 5%;" />

**Instruction-C**: *Please complete the method ${Method Name} within the following class **${Class Name}**. **${Class-level Info} ${Other Method Signatures} ${Target Method Contract Design}***

**${Class-level Info}** encompasses import statements, class name, class description, and class constructor, while **${Target Method Contract Design}** includes method signature, functional description, parameter/return description, and example input/output. Detailed content for each component can be found in Figure 2 of the paper.

#### LLMs with FIM Ability

The prompt of these models is the code context without any instruction: (i) for holistic generation, the
prompt is just the class skeleton; (ii) for incremental generation, the prompt in each iteration includes the class-level information, generated methods, and the target method contract design; (iii) for
compositional generation, the prompt for each method includes the class-level information, other method signatures, and the target method contract design. 

To stimulate FIM ability, we've included a placeholder <insert> where method bodies need to be generated. Here's an example for holistic generation:

```python
${Class-level Info}
${Method_1 Contract Design}
<insert>
${Method_2 Contract Design}
<insert>
...
${Method_N Contract Design}
<insert>
```

#### with Neither IF nor FIM Ability
These models utilize input similar to LLMs with FIM ability but without any <insert> placeholders.

