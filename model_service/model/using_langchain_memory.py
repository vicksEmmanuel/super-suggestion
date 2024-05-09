import torch
import argparse
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from model.lm_eval.utils import build_fim_sentinel_dict, self_infill_split
from model.code_generator_pipeline import CodeGeneratorPipeline


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    StoppingCriteriaList,
    LogitsProcessorList
)
from model.lm_eval.generation_pipelines.self_infill_utils import (
    SelfInfillingLogitsProcessor,
    SelfInfillEndOfFunctionCriteria,
)

def setup_generation_config(
        tokenizer, 
        suffix_prompt, 
        stop_words,
        args,
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
                tau=args.self_infill_tau,
                suffix_prompt=suffix_prompt
            )
        ])
    }
    tokenizer_kwargs = {
        "return_token_type_ids": False,
    }
    return gen_kwargs, tokenizer_kwargs


def using_lanchain_memory(args):

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, 
        trust_remote_code=True, 
        use_fast=False
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        trust_remote_code=True, 
        # device_map = "auto"
    ).to(args.device)

    # Read provided prompt from a file
    prompt = f"""<code>{args.prompt}"""
    suffix_prompt = args.suffix_prompt.strip() if args.suffix_prompt else ""
    
    pipeline = CodeGeneratorPipeline(
        model=model,
        tokenizer=tokenizer,
        self_infill_tau = args.self_infill_tau
    )
    
    # Create a ConversationBufferMemory for context-aware generation
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        input_key="prompt"
    )

    code = pipeline(prompt, suffix_prompt)

    # Update the memory with the user's input
    memory.save_context({"prompt": prompt}, {"code": code["code"]})


    prefix = code["prefix"]
    infill = code["infill"]
    suffix = code["suffix"]
    code = code["code"]

    return {
        "prefix": prefix,
        "infill": infill,
        "suffix": suffix,
        "code": code
    }


class  GenerateCodeModel:
    def __init__(self, prompt, suffix_prompt):
        self.model = 'codellama/CodeLlama-7b-hf'
        self.device = "cpu"
        self.self_infill_tau = 0.25
        self.prompt = prompt
        self.suffix_prompt = suffix_prompt

    def generate_code(self):
        parser = argparse.ArgumentParser(description='Generate code using the specified model.')
        parser.add_argument('--model', type=str, default=self.model, help='Model to use for code generation.')
        parser.add_argument('--device', type=str, default=self.device, help='Device to run the model on.')
        parser.add_argument('--self_infill_tau', type=float, default=self.self_infill_tau, help='Self-infill temperature.')
        parser.add_argument('--prompt', type=str, default=self.prompt, help='Prompt for code generation.')
        parser.add_argument('--suffix_prompt', type=str, default=self.suffix_prompt, help='Suffix prompt for code generation.')

        args = parser.parse_args()
        return using_lanchain_memory(args)