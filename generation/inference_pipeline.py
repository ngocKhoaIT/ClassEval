import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoModel
import json
from tqdm import tqdm
from inference_util import InferenceUtil, ModelName, GenerationStrategy
import prompt_strategies as ps
from prompt_strategies import PromptStyle
import os
import time
import openai
from importlib.util import find_spec

class InferencePipeline:

    def __init__(self, args):
        with open(args.data_path, 'r', encoding = 'utf-8') as f:
            self.file_cont = json.load(f)
        self.greedy = args.greedy
        self.output_path = args.output_path
        self.cuda = "cuda" if torch.cuda.is_available() else "cpu"
        if args.cuda is not None:
            if self.cuda != "cuda":
                raise RuntimeError("--cuda was provided, but this PyTorch installation cannot use CUDA")
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.cuda])
        if self.cuda == "cuda":
            print(f"Inference device: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})", flush=True)
        else:
            print("Inference device: CPU (CUDA unavailable in this PyTorch environment)", flush=True)
        self.generation_strategy = args.generation_strategy
        self.model_name = args.model
        self.checkpoint = args.checkpoint
        self.temperature = args.temperature
        self.max_length = args.max_length
        self.openai_key = args.openai_key
        self.openai_base = args.openai_base
        self.google_api_key = args.google_api_key
        self.prompt_style = args.prompt_style
        self.fewshot_examples = ""
        if self.prompt_style == PromptStyle.FewShot.value:
            # example tasks are shown in the prompt, so they are excluded from generation
            self.fewshot_examples = ps.build_fewshot_examples(self.file_cont, args.fewshot_ids)
            self.file_cont = [cont for cont in self.file_cont if cont['task_id'] not in args.fewshot_ids]
        if args.task_ids is not None:
            selected_ids = set(args.task_ids)
            available_ids = {cont['task_id'] for cont in self.file_cont}
            missing_ids = selected_ids - available_ids
            if missing_ids:
                raise ValueError(f"Unknown or excluded task IDs: {', '.join(sorted(missing_ids))}")
            self.file_cont = [cont for cont in self.file_cont if cont['task_id'] in selected_ids]

        self.get_model_tokenizer_and_config()
        self.SAMPLE_NUMS = 1 if self.greedy == 1 else args.sample
        self.do_sample = False if self.greedy == 1 else True

    def get_model_tokenizer_and_config(self):
        if self.model_name == ModelName.GPT_3_5.value or self.model_name == ModelName.GPT_4.value:
            return
        elif self.model_name == ModelName.Gemini_Pro.value:
            from google import genai
            from google.genai import types
            self.client = genai.Client(api_key = self.google_api_key)
            self.gemini_model = self.checkpoint if self.checkpoint.startswith("gemini") else "gemini-3.5-flash"
            safety_settings = [
                types.SafetySetting(category = category, threshold = "BLOCK_NONE")
                for category in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                                 "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
            ]

            # Set up the model
            self.gemini_config = types.GenerateContentConfig(
                temperature = 0 if self.greedy == 1 else self.temperature,
                top_p = 1,
                top_k = 1,
                max_output_tokens = self.max_length,
                safety_settings = safety_settings,
            )
        elif self.model_name == ModelName.ChatGLM.value:
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, trust_remote_code = True)
            load_args = {"trust_remote_code": True}
            if self.cuda == "cuda" and find_spec("accelerate") is not None:
                load_args["device_map"] = "auto"
            self.model = AutoModel.from_pretrained(self.checkpoint, **load_args)
            if self.cuda == "cuda":
                self.model = self.model.half()
            if "device_map" not in load_args:
                self.model = self.model.to(self.cuda)
            self.model = self.model.eval()
        elif self.model_name == ModelName.PolyCoder.value or self.model_name == ModelName.SantaCoder.value:
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, trust_remote_code = True)
            load_args = {"trust_remote_code": True, "torch_dtype": torch.float16 if self.cuda == "cuda" else torch.float32}
            if self.cuda == "cuda" and find_spec("accelerate") is not None:
                load_args["device_map"] = "auto"
            self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint, **load_args)
            if "device_map" not in load_args:
                self.model = self.model.to(self.cuda)
            self.model = self.model.eval()
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.generation_config = GenerationConfig(
                temperature = self.temperature,
                eos_token_id = self.tokenizer.eos_token_id,
                pad_token_id = self.tokenizer.pad_token_id,
                do_sample = True
            ) if self.greedy == 0 else GenerationConfig(
                eos_token_id = self.tokenizer.eos_token_id,
                pad_token_id = self.tokenizer.pad_token_id
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, trust_remote_code = True)
            load_args = {"trust_remote_code": True, "torch_dtype": torch.float16 if self.cuda == "cuda" else torch.float32}
            if self.cuda == "cuda" and find_spec("accelerate") is not None:
                load_args["device_map"] = "auto"
            self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint, **load_args)
            if "device_map" not in load_args:
                self.model = self.model.to(self.cuda)
            self.model = self.model.eval()
            self.generation_config = GenerationConfig(
                temperature = self.temperature,
                eos_token_id = self.tokenizer.eos_token_id,
                pad_token_id = self.tokenizer.pad_token_id,
                do_sample = True
            ) if self.greedy == 0 else GenerationConfig(
                eos_token_id = self.tokenizer.eos_token_id,
                pad_token_id = self.tokenizer.pad_token_id
            )

    def save_result(self, result):
        with open(self.output_path, 'w', encoding = 'utf-8') as f:
            json.dump(result, f, indent=4)

    @staticmethod
    def _retry_on_transient(func, max_retries=8, base_delay=15):
        """Retry a function call on transient API errors (429, 503, etc.).
        Default schedule: 15s, 30s, 60s, 120s, 240s, 480s, 960s, 1920s (~53 min total)."""
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                err_str = str(e)
                is_transient = any(code in err_str for code in ["429", "503", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "rate limit", "overloaded", "high demand"])
                if is_transient and attempt < max_retries - 1:
                    # Check if the API suggests a specific retry delay
                    import re as _re
                    retry_match = _re.search(r'[Rr]etry in (\d+(?:\.\d+)?)s', err_str)
                    delay = float(retry_match.group(1)) + 2 if retry_match else base_delay * (2 ** attempt)
                    print(f"  [retry {attempt+1}/{max_retries}] {err_str[:120]}... waiting {delay:.0f}s")
                    time.sleep(delay)
                else:
                    raise

    def model_generate(self, prompt):
        if self.model_name == ModelName.GPT_3_5.value or self.model_name == ModelName.GPT_4.value:
            openai.api_key = self.openai_key
            openai.api_base = self.openai_base
            model_id = "gpt-3.5-turbo" if self.model_name == ModelName.GPT_3_5.value else "gpt-4"
            def _call_openai():
                return openai.ChatCompletion.create(
                    max_tokens=self.max_length,
                    temperature=0 if self.greedy == 1 else self.temperature,
                    model=model_id,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
            response = self._retry_on_transient(_call_openai)
            outputs = response.choices[0]["message"]["content"]
        elif self.model_name == ModelName.Gemini_Pro.value:
            def _call_gemini():
                return self.client.models.generate_content(model = self.gemini_model, contents = prompt,
                                                           config = self.gemini_config)
            response = self._retry_on_transient(_call_gemini)
            outputs = response.text or ""
        elif self.model_name == ModelName.DeepSeekCoder_inst.value:
            messages=[
                { "role": "user", "content": prompt }
            ]
            # Leave room for a completion and decode only newly generated tokens.
            # DeepSeek-Coder's chat template includes the whole prompt in generate()'s
            # returned sequence, so decoding it all leaks the prompt into evaluation.
            # The two solved examples make Few-shot prompts longer than 2,000 tokens.
            # Keep the full prompt and treat max_length as the completion budget here.
            # The selected 1.3B checkpoint has a 16K context window.
            max_new_tokens = min(1024, max(256, self.max_length // 4))
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt = True, return_tensors = "pt"
            ).to(self.cuda)
            eot_id = self.tokenizer.convert_tokens_to_ids("<|EOT|>")
            eos_ids = [self.tokenizer.eos_token_id]
            if eot_id is not None and eot_id not in eos_ids:
                eos_ids.append(eot_id)
            outputs = self.model.generate(
                input_ids, generation_config = self.generation_config,
                max_new_tokens = max_new_tokens, do_sample = self.do_sample,
                eos_token_id = eos_ids
            )
            outputs = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens = True)
        elif self.model_name == ModelName.ChatGLM.value:
            outputs, _ = self.model.chat(self.tokenizer, prompt, temperature = self.temperature, do_sample = self.do_sample)
        else:
            if self.model_name == ModelName.SantaCoder.value and self.prompt_style != PromptStyle.Baseline.value:
                # Keep the end of the task prompt (especially its target skeleton) when
                # long few-shot examples need truncation, and reserve the remaining
                # context window for newly generated tokens.
                original_truncation_side = self.tokenizer.truncation_side
                self.tokenizer.truncation_side = "left"
                try:
                    input_ids = self.tokenizer.encode(
                        prompt, return_tensors = "pt", max_length = self.max_length, truncation = True
                    ).to(self.cuda)
                finally:
                    self.tokenizer.truncation_side = original_truncation_side
                max_new_tokens = max(1, self.max_length - input_ids.shape[-1])
                outputs = self.model.generate(
                    input_ids, generation_config = self.generation_config,
                    max_new_tokens = max_new_tokens, do_sample = self.do_sample
                )
                outputs = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens = True)
            else:
                input_ids = self.tokenizer.encode(prompt, return_tensors = "pt", max_length = self.max_length, truncation = True).to(self.cuda)
                outputs = self.model.generate(input_ids, generation_config = self.generation_config, 
                                                max_length = self.max_length, do_sample = self.do_sample)
                outputs = self.tokenizer.decode(outputs[0], skip_special_tokens = True)
        return outputs

    def plan_then_code(self, cont):
        if self.model_name == ModelName.SantaCoder.value:
            plan_prompt = ps.santacoder_plan_instruction(cont)
            plan = self.model_generate(plan_prompt).strip()
            code_prompt = ps.santacoder_code_from_plan_instruction(cont, plan)
            code = self.model_generate(code_prompt)
            return cont['skeleton'].rstrip() + "\n" + code
        plan_prompt = InferenceUtil.generate_prompt(ps.plan_instruction(cont), self.model_name)
        plan = ps.strip_response(self.model_generate(plan_prompt))
        code_prompt = InferenceUtil.generate_prompt(ps.code_from_plan_instruction(cont, plan), self.model_name)
        code = ps.strip_response(self.model_generate(code_prompt))
        return "### Plan\n" + plan + "\n\n### Code\n" + code

    def construct_prompt(self, strategy, info):
        prompt = ""
        if strategy == GenerationStrategy.Holistic:
            if self.model_name == ModelName.PolyCoder.value or self.model_name == ModelName.SantaCoder.value:
                skeleton = info['skeleton']
                if self.model_name == ModelName.SantaCoder.value:
                    if self.prompt_style == PromptStyle.CoT.value:
                        prompt = ps.santacoder_cot_instruction(info)
                    elif self.prompt_style == PromptStyle.FewShot.value:
                        prompt = ps.santacoder_fewshot_instruction(info, self.fewshot_examples)
                    else:
                        prompt = skeleton
                else:
                    prompt = skeleton
            else:
                class_name = info['class_name']
                skeleton = info['skeleton']
                if self.prompt_style == PromptStyle.CoT.value:
                    instruction = ps.cot_instruction(info)
                elif self.prompt_style == PromptStyle.FewShot.value:
                    instruction = ps.fewshot_instruction(info, self.fewshot_examples)
                else:
                    instruction = f"Please complete the class {class_name} in the following code."
                    instruction = instruction + '\n' + skeleton
                prompt = InferenceUtil.generate_prompt(instruction, self.model_name)

        elif strategy == GenerationStrategy.Incremental:
            if self.model_name == ModelName.PolyCoder.value or self.model_name == ModelName.SantaCoder.value:
                prompt = info['skeleton']
            else:
                prompt = info['instruction'] + info['skeleton']
                prompt = InferenceUtil.generate_prompt(prompt, self.model_name)

        elif strategy == GenerationStrategy.Compositional:
            if self.model_name == ModelName.PolyCoder.value or self.model_name == ModelName.SantaCoder.value:
                prompt = info['skeleton']
            else:
                prompt = info['instruction'] + info['skeleton']
                prompt = InferenceUtil.generate_prompt(prompt, self.model_name)

        return prompt

    def post_process(self, result):
        if self.generation_strategy == GenerationStrategy.Incremental.value:
            for cont in result:
                pred = []
                for result in cont['predict']:
                    pred.append(result[-1])
                cont['predict'] = pred
        elif self.generation_strategy == GenerationStrategy.Compositional.value:
            for cont in result:
                cont['raw_output'] = cont['predict'].copy()
            for cont in result:
                cont['predict'] = []
                for raw_output in cont['raw_output']:
                    class_code = '\n'.join(cont['import_statement']) + '\n' + cont['class_constructor']
                    for i in range(len(raw_output)):
                        method_name = cont['methods_info'][i]['method_name']
                        code = raw_output[i]
                        method_code = InferenceUtil.extract_method_code(code, method_name)
                        class_code += '\n\n' + method_code
                    cont['predict'].append(class_code)

    def _load_existing_results(self):
        """Load previously saved results for resume support."""
        if os.path.exists(self.output_path):
            try:
                with open(self.output_path, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                done_ids = {cont['task_id'] for cont in existing if 'predict' in cont}
                print(f"  [resume] Found {len(done_ids)} completed tasks in {self.output_path}")
                return existing, done_ids
            except (json.JSONDecodeError, KeyError):
                pass
        return [], set()

    def pipeline(self):
        error_task_id_list = []
        if self.generation_strategy == GenerationStrategy.Holistic.value:
            result, done_ids = self._load_existing_results()
            for cont in tqdm(self.file_cont):
                if cont['task_id'] in done_ids:
                    continue
                pred = []
                raw_output = []
                try:
                    for _ in range(self.SAMPLE_NUMS):
                        if self.prompt_style == PromptStyle.PlanThenCode.value:
                            outputs = self.plan_then_code(cont)
                        else:
                            prompt = self.construct_prompt(GenerationStrategy.Holistic, cont)
                            outputs = self.model_generate(prompt)
                            if self.model_name == ModelName.SantaCoder.value and self.prompt_style != PromptStyle.Baseline.value:
                                outputs = cont['skeleton'].rstrip() + "\n" + outputs
                        raw_output.append(outputs)
                        if self.prompt_style != PromptStyle.Baseline.value:
                            outputs = ps.extract_final_code(outputs, cont['class_name'])
                        pred.append(outputs)
                    cont['predict'] = pred
                    if self.prompt_style != PromptStyle.Baseline.value:
                        cont['raw_output'] = raw_output
                    result.append(cont)
                    self.save_result(result)

                except Exception as e:
                    print(e)
                    print("IDX: ", cont['task_id'])
                    error_task_id_list.append(cont['task_id'])

        elif self.generation_strategy == GenerationStrategy.Incremental.value:
            result = []
            for cont in tqdm(self.file_cont):
                cont['predict'] = []
                cont['raw_output'] = []
                for _ in range(self.SAMPLE_NUMS):
                    pred = []
                    raw_output = []
                    try:
                        class_name = cont['class_name']
                        methods_info = cont['methods_info']
                        imports = '\n'.join(cont['import_statement'])
                        class_init = InferenceUtil.add_desc_to_init(cont['class_description'], cont['class_constructor'])
                        class_text = imports + '\n' + class_init
                        for method in methods_info:
                            # construct prompt
                            method_name = method['method_name']
                            inst = f"please complete {method_name} method in the following class {class_name}\n\n"
                            class_text_desc = class_text + "\n\n    " + method['method_description']
                            prompt = self.construct_prompt(GenerationStrategy.Incremental, {"instruction":inst, "skeleton": class_text_desc})

                            # generate model output
                            outputs = self.model_generate(prompt)
                            raw_output.append(outputs)

                            # extract valid generated code
                            generated_method_code = InferenceUtil.extract_method_code(outputs, method_name)
                            class_text += '\n\n' + generated_method_code
                            pred.append(class_text)

                        cont['predict'].append(pred)
                        cont['raw_output'].append(raw_output)
                        
                    except Exception as e:
                        print(e)
                        print("IDX: ", cont['task_id'])
                        error_task_id_list.append(cont['task_id'])

                result.append(cont)
                self.save_result(result)

        elif self.generation_strategy == GenerationStrategy.Compositional.value:
            result = []
            for cont in tqdm(self.file_cont):
                cont['predict'] = []
                for _ in range(self.SAMPLE_NUMS):
                    pred = []
                    try:
                        class_name = cont['class_name']
                        methods_info = cont['methods_info']
                        imports = '\n'.join(cont['import_statement'])
                        class_init = InferenceUtil.add_desc_to_init(cont['class_description'], cont['class_constructor'])
                        for method_to_generate in methods_info:
                            class_text = imports + '\n' + class_init
                            # gather each method's signature to consruct class level skeleton
                            for method in methods_info:
                                if method['method_name'] == method_to_generate['method_name']:
                                    continue
                                class_text += InferenceUtil.get_method_signature(method['method_description'], method['method_name']) + "\n        pass\n\n"
                            # construct prompt
                            method_name = method_to_generate['method_name']
                            inst = f"please complete {method_name} method in the following class {class_name}\n\n"
                            class_text_desc = class_text + "\n\n    " + method_to_generate['method_description']
                            prompt = self.construct_prompt(GenerationStrategy.Compositional, {"instruction":inst, "skeleton": class_text_desc})

                            # generate model output
                            outputs = self.model_generate(prompt)
                            pred.append(outputs)

                        cont['predict'].append(pred)
                        
                    except Exception as e:
                        print(e)
                        print("IDX: ", cont['task_id'])
                        error_task_id_list.append(cont['task_id'])

                result.append(cont)
                self.save_result(result)
        else:
            print("Unknown Generation Strategy")
            return
        
        print("error_task_id_list: ", error_task_id_list)
        self.post_process(result)
        self.save_result(result)
