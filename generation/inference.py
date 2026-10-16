import argparse
import os
from pathlib import Path

# Set this before importing Transformers/Hugging Face through inference_pipeline.
# The default user cache may be read-only on managed Windows installations.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HF_HOME = PROJECT_ROOT / "model_cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_HF_HOME / "hub"))

from inference_pipeline import InferencePipeline

def load_project_env():
    """Load simple KEY=value entries from the repository's private .env file."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not key.isidentifier():
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(key, value)

def args_init():
    load_project_env()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="../data/ClassEval_data.json",
        help="ClassEval data",
    )
    parser.add_argument(
        "--task_ids",
        nargs="+",
        default=None,
        help="Generate only these ClassEval task IDs (default: all tasks)",
    )
    parser.add_argument(
        "--greedy",
        type=int,
        default=1,
        help="Whether to generate model results with greedy strategy",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="model_output.json",
        help="output file path",
    )
    parser.add_argument(
        "--cuda",
        type=int,
        nargs="+",  # Accept one or more integers
        default=None,
        help="List of CUDA device(s), default value is None. If not set, use all available devices.",
    )
    parser.add_argument(
        "--generation_strategy",
        type=int,
        default=0,
        help="Holistic = 0, Incremental = 1, Compositional = 2",
    )
    parser.add_argument(
        "--model",
        type=int,
        default=1,
        help="Instruct_CodeGen = 0, WizardCoder = 1, Instruct_StarCoder = 2, InCoder = 3, \
        PolyCoder = 4, SantaCoder = 5, Vicuna = 6, ChatGLM = 7, GPT_3_5 = 8, GPT_4 = 9, others = 10, \
        Magicoder = 11, CodeGeeX2 = 12, DeepSeekCoder_inst = 13, Gemini_Pro = 14, CodeLlama_13b_inst = 15",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="WizardLM/WizardCoder-15B-V1.0",
        help="checkpoint of the model",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="temperature value in generation config",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="max length of model's generation result",
    )
    parser.add_argument(
        "--openai_key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY"),
        help="OpenAI API key (defaults to OPENAI_API_KEY from environment or .env)",
    )
    parser.add_argument(
        "--openai_base",
        type=str,
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI API base URL (defaults to OPENAI_BASE_URL from environment or .env)",
    )

    parser.add_argument(
        "--google_api_key",
        type=str,
        default=os.environ.get("GOOGLE_API_KEY"),
        help="Google API key (defaults to GOOGLE_API_KEY from environment or .env)",
    )

    parser.add_argument(
        "--prompt_style",
        type=int,
        default=0,
        help="Only for Holistic generation. Baseline = 0, CoT = 1, FewShot = 2, PlanThenCode = 3",
    )
    parser.add_argument(
        "--fewshot_ids",
        type=str,
        nargs="+",
        default=["ClassEval_6", "ClassEval_30"],
        help="task_ids used as few-shot examples (excluded from generation) when prompt_style = 2",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=5,
        help="The number of code samples that are randomly generated for each task.",
    )
    args = parser.parse_args()
    if args.prompt_style != 0 and args.generation_strategy != 0:
        parser.error("--prompt_style is only supported with --generation_strategy 0 (Holistic)")
    if args.model in (8, 9) and not args.openai_key:
        parser.error("Set OPENAI_API_KEY in .env or pass --openai_key for GPT models")
    if args.model == 14 and not args.google_api_key:
        parser.error("Set GOOGLE_API_KEY in .env or pass --google_api_key for Gemini")
    return args

if __name__ == '__main__':

    args = args_init()
    infer = InferencePipeline(args)
    infer.pipeline()
