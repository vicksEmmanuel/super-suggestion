import torch
import argparse
import sys
import os
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from code_generator_pipeline import CodeGeneratorPipeline
from lm_eval.utils import build_fim_sentinel_dict, self_infill_split

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    StoppingCriteriaList,
    LogitsProcessorList
)
from lm_eval.generation_pipelines.self_infill_utils import (
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

    print(f"Args {args.prompt_file}")

    # Read provided prompt from a file
    prompt = args.prompt_file
    suffix_prompt = args.suffix_prompt_file.strip() if args.suffix_prompt_file else ""
    
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


    code = code["code"]
    prefix = code["prefix"]
    infill = code["infill"]
    suffix = code["suffix"]

    return {
        "prefix": prefix,
        "infill": infill,
        "suffix": suffix,
        "code": code
    }



class  GenerateCodeModel:
    def __init__(self,prompt, suffix_prompt):
        self.model = 'codellama/CodeLlama-7b-hf'
        self.device = "cpu"
        self.self_infill_tau = 0.25
        self.prompt = prompt
        self.suffix_prompt = suffix_prompt

    def generate_code(self):
        return using_lanchain_memory(self)

if __name__ == "__main__":
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Self-infilling code generation')
    parser.add_argument('--model', type=str, default="codellama/CodeLlama-7b-hf", help='HF Model name or path')
    parser.add_argument('--device', type=str, default="cuda", help='Device to use (cuda or cpu)')
    parser.add_argument('--self-infill-tau', type=float, default=0.25, help='threshold tau for self-infilling interruption')
    parser.add_argument('--prompt-file', type=str, default=None, help='File path for the user prompt')
    parser.add_argument('--suffix-prompt-file', type=str, default=None, help='File path for the custom suffix prompt')
    args = parser.parse_args()

    # Read provided prompt from a file
    if args.prompt_file:
        try:
            with open(args.prompt_file, "r") as f:
                args.prompt = f.read()
        except IOError:
            print(f"Error: Could not read file {args.prompt_file}")
            sys.exit(1)
    else:
        print("No user prompt provided. Exiting.")
        sys.exit(1)

    # Read provided suffix prompt from the specified file
    if args.suffix_prompt_file:
        try:
            with open(args.suffix_prompt_file, "r") as f:
                args.suffix_prompt = f.read().strip()
        except IOError:
            print(f"Error: Could not read file {args.suffix_prompt_file}")
            sys.exit(1)
    else:
        print("No user prompt provided. Defaults to empty.")
        args.suffix_prompt = ""


    using_lanchain_memory(args)