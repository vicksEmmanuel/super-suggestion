import torch
import argparse
import sys
import os
from typing import Any, List, Mapping, Optional
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteriaList,
    LogitsProcessorList,
    Pipeline,
)
from model.lm_eval.generation_pipelines.self_infill_utils import (
    SelfInfillingLogitsProcessor,
    SelfInfillEndOfFunctionCriteria,
)

from model.lm_eval.utils import (
    build_fim_sentinel_dict,
    self_infill_split,
)

class CodeGeneratorPipeline(Pipeline):

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        self_infill_tau: str,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(tokenizer=tokenizer, model=model, *args, **kwargs)
        self.model = model
        self.tokenizer = tokenizer
        self.self_infill_tau = self_infill_tau

    def _sanitize_parameters(self, **pipeline_parameters):
        preprocess_kwargs = {}
        if "model" in pipeline_parameters:
            preprocess_kwargs["model"] = pipeline_parameters["model"]
        if "prompt" in pipeline_parameters:
            preprocess_kwargs["prompt"] = pipeline_parameters["prompt"]
        if "suffix_prompt" in pipeline_parameters:
            preprocess_kwargs["suffix_prompt"] = pipeline_parameters["suffix_prompt"]

        return preprocess_kwargs, {}, {}
    
    def _forward(self, inputs, **args):
        si_inputs, gen_kwargs = inputs

        si_generated_tokens = self.model.generate(
            input_ids=si_inputs["input_ids"], 
            attention_mask=si_inputs["attention_mask"],
            **gen_kwargs,
        )
        return si_generated_tokens

    
    def preprocess(self, prompt: str, suffix_prompt: str = ""):
        # Setup infilling sentinel tokens
        fim_sentinel_dict = build_fim_sentinel_dict(self.tokenizer)
        fim_prefix = fim_sentinel_dict["fim_prefix"]
        si_prompts = f"{fim_prefix}{prompt}"

        # Setup stop words
        stop_words = [
            "</code>",
            self.tokenizer.eos_token,
        ]
        if isinstance(fim_sentinel_dict["fim_ending"], list):
            stop_words += fim_sentinel_dict["fim_ending"]
        else:
            stop_words.append(fim_sentinel_dict["fim_ending"])

        gen_kwargs, tokenizer_kwargs = setup_generation_config(
            self.tokenizer,
            suffix_prompt,
            stop_words,
            self.self_infill_tau,
        )

        # Tokenize inputs
        si_inputs = self.tokenizer(
            si_prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            **tokenizer_kwargs,
        ).to(self.model.device)

        return si_inputs, gen_kwargs

    def postprocess(self, model_outputs):
        si_generated_tokens = model_outputs
        si_generated_code = self.tokenizer.batch_decode(
            si_generated_tokens,
            skip_special_tokens=False, 
            clean_up_tokenization_spaces=False
        )[0]
        print(f">>> Raw self-infilled code:\n\n ======== \n\n {si_generated_code} \n\n ========== \n\n")
        fim_sentinel_dict = build_fim_sentinel_dict(self.tokenizer)
        # Parse self-infilled generation to code
        result_dict = self_infill_split(self.tokenizer, si_generated_code, fim_sentinel_dict)
        prefix, infill, suffix = result_dict["split"]

        code = prefix + infill + suffix

        if result_dict["fim_suffix_present"]:
            print("[self-infill] self-infilling interruption invoked")
            if result_dict["fim_middle_present"] and result_dict["fim_ending_present"]:
                print("[self-infill] successfully self-infilled")
            elif result_dict["fim_middle_present"] and not result_dict["fim_ending_present"]:
                print("[self-infill] self-infilling fails to join the suffix")
            elif not result_dict["fim_middle_present"]:
                print("[self-infill] self-infilling fails to produce a suffix")
            else:
                print("[self-infill] should not happen here")
        else:
            print("[self-infill] NOT invoking self-infilling")

        print(">>> Final code:\n", code)

        return {
            "code": code,
            "prefix": prefix,
            "infill": infill,
            "suffix": suffix
        }
    

def setup_generation_config(
        tokenizer, 
        suffix_prompt, 
        stop_words,
        self_infill_tau,
    ):
    """
    Setup configuration for generate.
    """
    gen_kwargs = {
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.95,
        "max_new_tokens": 128,
        "stopping_criteria": StoppingCriteriaList([
            SelfInfillEndOfFunctionCriteria(0, stop_words, tokenizer)
        ]),
        "logits_processor": LogitsProcessorList([
            SelfInfillingLogitsProcessor(
                0, stop_words, tokenizer, 
                tau=self_infill_tau,
                suffix_prompt=suffix_prompt
            )
        ])
    }
    tokenizer_kwargs = {
        "return_token_type_ids": False,
    }
    return gen_kwargs, tokenizer_kwargs
