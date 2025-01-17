import torch
from transformers import (
    StoppingCriteria, 
)
# the following code is adapted from 
# https://github.com/liftoff/pyminifier/blob/master/pyminifier/minification.py
# Import built-in modules
import tokenize
import io
import zlib

def remove_blank_lines(source):
    """
    Removes blank lines from *source* and returns the result.

    Example:

    .. code-block:: python

        test = "foo"

        test2 = "bar"

    Will become:

    .. code-block:: python

        test = "foo"
        test2 = "bar"
    """
    io_obj = io.StringIO(source)
    source = [a for a in io_obj.readlines() if a.strip()]
    return "".join(source)

def remove_comments_and_docstrings(source):
    """
    Returns *source* minus comments and docstrings.

    .. note:: Uses Python's built-in tokenize module to great effect.

    Example::

        def noop(): # This is a comment
            '''
            Does nothing.
            '''
            pass # Don't do anything

    Will become::

        def noop():
            pass
    """
    io_obj = io.StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += (" " * (start_col - last_col))
        # Remove comments:
        if token_type == tokenize.COMMENT:
            pass
        # This series of conditionals removes docstrings:
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
        # This is likely a docstring; double-check we're not inside an operator:
                if prev_toktype != tokenize.NEWLINE:
                    # Note regarding NEWLINE vs NL: The tokenize module
                    # differentiates between newlines that start a new statement
                    # and newlines inside of operators such as parens, brackes,
                    # and curly braces.  Newlines inside of operators are
                    # NEWLINE and newlines that start new code are NL.
                    # Catch whole-module docstrings:
                    if start_col > 0:
                        # Unlabelled indentation means we're inside an operator
                        out += token_string
                    # Note regarding the INDENT token: The tokenize module does
                    # not label indentation inside of an operator (parens,
                    # brackets, and curly braces) as actual indentation.
                    # For example:
                    # def foo():
                    #     "The spaces before this docstring are tokenize.INDENT"
                    #     test = [
                    #         "The spaces before this string do not get a token"
                    #     ]
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    return out

# https://github.com/facebookresearch/coder_reviewer_reranking/blob/main/sample_selectors.py#L342
import zlib

def check_degenerate(code):
    try:
        code = remove_comments_and_docstrings(code)
        code = remove_blank_lines(code)
    except:
        code = ""
    return code.strip() in ["", "pass", "return"]

def check_repeat(code, threshold=0.1):
    bytes_code = bytes(code, encoding="utf-8")
    if len(bytes_code) == 0:
        return True
    comp_code = zlib.compress(bytes_code)
    return len(comp_code) / len(bytes_code) < threshold

class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""
    def __init__(self, start_length, eof_strings, tokenizer, check_fn=None):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer
        if check_fn is None:
            check_fn = lambda decoded_generation: any(
                [stop_string in decoded_generation for stop_string in self.eof_strings]
            )
        self.check_fn = check_fn

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(input_ids[:, self.start_length :])
        return all([self.check_fn(decoded_generation) for decoded_generation in decoded_generations])

class TooLongFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if the generated function is too long by a certain multiplier based on input length."""

    def __init__(self, input_length, multiplier):
        self.input_length = input_length
        self.multiplier = multiplier

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if generated sequence is too long."""
        return input_ids.shape[1] > int(self.input_length * self.multiplier)

def compact_lines(lines):
    merged_lines = []
    leading_nonnewlines = -1
    temp_list = []
    # find the first line that is not \n
    for i, sol_line in enumerate(lines):
        if sol_line == "":
            temp_list.append(sol_line)
        else:
            leading_nonnewlines = i
            temp_list.append(sol_line)
            break
    # if no not-newline is found, return
    if leading_nonnewlines == -1:
        return ["\n".join(lines)]
    lines[leading_nonnewlines] = "\n".join(temp_list)
    temp = ""
    for i, sol_line in enumerate(lines[leading_nonnewlines:]):
        if sol_line == "":
            temp = temp + "\n"
        else:
            if temp:
                merged_lines.append(temp)
            temp = sol_line
    if temp:
        # split the last line into line + \n
        merged_lines.append(temp)
    return merged_lines

def build_fim_sentinel_dict(tokenizer, suffix_first=False):
    model_id = tokenizer.name_or_path
    if model_id.startswith("codellama/CodeLlama"):

        print(f"Tokenizer: Prefix= {tokenizer.prefix_token} , Surfix= {tokenizer.suffix_token} , Middle= {tokenizer.middle_token}")

        fim_prefix = tokenizer.prefix_token
        fim_suffix = tokenizer.suffix_token
        fim_middle = tokenizer.middle_token
        # we add a space after prefix token, to ensure that
        # the tokenizer still encodes the prompt as if there are leading spaces
        # see Codellama paper for more details
        if suffix_first:
            fim_middle = fim_middle + " "
        else:
            fim_prefix = fim_prefix + " "

        print(f"EOT_TOKEN: {tokenizer.eot_token} \n PREFIX_ID: {tokenizer.prefix_id} \n SURFIX_ID: {tokenizer.suffix_id} \n MIDDLE_ID: {tokenizer.middle_id} \n EOT_ID: {tokenizer.eot_id} ")
        
        fim_ending = tokenizer.eot_token
        fim_prefix_id = tokenizer.prefix_id
        fim_suffix_id = tokenizer.suffix_id
        fim_middle_id = tokenizer.middle_id
        fim_ending_id = tokenizer.eot_id
    elif "deepseek" in model_id:
        fim_prefix = "<｜fim▁begin｜>"
        fim_suffix = "<｜fim▁hole｜>"
        fim_middle = "<｜fim▁end｜>"
        fim_ending = "<｜end▁of▁sentence｜>"
        fim_prefix_id = tokenizer.convert_tokens_to_ids(fim_prefix)
        fim_suffix_id = tokenizer.convert_tokens_to_ids(fim_suffix)
        fim_middle_id = tokenizer.convert_tokens_to_ids(fim_middle)
        fim_ending_id = tokenizer.eos_token_id
    elif "codegen" in model_id:
        # codegen
        fim_prefix = ""
        fim_suffix = "<mask_1>"
        fim_middle = "<|endoftext|>" + "<sep>" + "<mask_1>"
        fim_ending = "<eom>"
        fim_prefix_id = None
        fim_middle_id = None
        fim_suffix_id = None
        fim_ending_id = None
    elif "incoder" in model_id:
        # incoder
        fim_prefix = ""
        fim_suffix = "<|mask:0|>"
        fim_middle = "<|mask:1|>" + "<|mask:0|>"
        fim_ending = "<|endofmask|>"
        fim_prefix_id = None
        fim_middle_id = None
        fim_suffix_id = None
        fim_ending_id = None
    elif "codegemma" in model_id:
        fim_prefix = "<|fim_prefix|>"
        fim_suffix = "<|fim_suffix|>"
        fim_middle = "<|fim_middle|>"
        fim_ending = [tokenizer.eos_token, '<|file_separator|>']
        fim_prefix_id = tokenizer.convert_tokens_to_ids(fim_prefix)
        fim_suffix_id = tokenizer.convert_tokens_to_ids(fim_suffix)
        fim_middle_id = tokenizer.convert_tokens_to_ids(fim_middle)
        fim_ending_id = [
            tokenizer.convert_tokens_to_ids('<|file_separator|>'),
            tokenizer.eos_token_id
        ]
    elif "starcoder2" in model_id:
        fim_prefix = "<fim_prefix>"
        fim_suffix = "<fim_suffix>"
        fim_middle = "<fim_middle>"
        fim_ending = ["<|endoftext|>", "<file_sep>", "<repo_name>"]
        fim_prefix_id = tokenizer.convert_tokens_to_ids(fim_prefix)
        fim_suffix_id = tokenizer.convert_tokens_to_ids(fim_suffix)
        fim_middle_id = tokenizer.convert_tokens_to_ids(fim_middle)
        fim_ending_id = [
            tokenizer.convert_tokens_to_ids('<file_sep>'),
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids('<repo_name>')
        ]
    else:
        if model_id in ["bigcode/santacoder"]:
            fim_prefix = "<fim-prefix>"
            fim_suffix = "<fim-suffix>"
            fim_middle = "<fim-middle>"
            fim_ending = "<|endoftext|>"
        elif model_id.startswith("bigcode/starcoder"):
            fim_prefix = "<fim_prefix>"
            fim_suffix = "<fim_suffix>"
            fim_middle = "<fim_middle>"
            fim_ending = "<|endoftext|>"
        elif model_id == "WizardLM/WizardCoder-15B-V1.0":
            fim_prefix = "<fim_prefix>"
            fim_suffix = "<fim_suffix>"
            fim_middle = "<fim_middle>"
            fim_ending = "<|endoftext|>"
        else:
            raise ValueError(f"Self-infilling not yet supported for: {model_id}")
        fim_prefix_id = tokenizer.convert_tokens_to_ids(fim_prefix)
        fim_suffix_id = tokenizer.convert_tokens_to_ids(fim_suffix)
        fim_middle_id = tokenizer.convert_tokens_to_ids(fim_middle)
        fim_ending_id = tokenizer.eos_token_id
    return {
        "fim_prefix": fim_prefix,
        "fim_prefix_id": fim_prefix_id,
        "fim_suffix": fim_suffix,
        "fim_suffix_id": fim_suffix_id,
        "fim_middle": fim_middle,
        "fim_middle_id": fim_middle_id,
        "fim_ending": fim_ending,
        "fim_ending_id": fim_ending_id,
    }

def extract_sp_dsonek(prompt):
    """
        This function extracts the default suffix prompt 
        from the input of DS-1000 problems
    """
    fallback_suffix_prompt = "\n</code>\n"
    if "BEGIN SOLUTION\n<code>" in prompt:
        # extract variables
        prompt_lines = prompt.split("\n")
        line_of_interest_idx = -1
        for i, prompt_line in enumerate(prompt_lines):
            if prompt_line.startswith("BEGIN SOLUTION"):
                line_of_interest_idx = i - 1
                break
        line_of_interest = prompt_lines[line_of_interest_idx]
        if line_of_interest.startswith("\n</code>"):
            default_suffix_prompt = fallback_suffix_prompt
        else:
            # we extract the mentioned variable and
            # shape the generation s.t. it must hit the variable.
            last_quote_index = line_of_interest.rfind("`")
            
            # If a single-quote is found, extract content within it
            if last_quote_index != -1:
                # Find the index of the first single-quote from the end of the string
                first_quote_index = line_of_interest.rfind("`", 0, last_quote_index)
                
                # Extract content within single-quotes
                if first_quote_index != -1:
                    default_suffix_prompt = line_of_interest[first_quote_index + 1:last_quote_index]
                else:
                    default_suffix_prompt = fallback_suffix_prompt
            else:
                expr_idx = line_of_interest.find("=")
                if expr_idx == -1:
                    # there is not a assignment neither. use fall-back
                    default_suffix_prompt = fallback_suffix_prompt
                else:
                    default_suffix_prompt = line_of_interest[:expr_idx].rstrip()
    elif "# SOLUTION START" in prompt:
        # Matplotlib questions.
        default_suffix_prompt = "\n# SOLUTION END\n"
    elif "### BEGIN SOLUTION" in prompt:
        # some tasks are directed to complete a function.
        default_suffix_prompt = "return"
    else:
        default_suffix_prompt = fallback_suffix_prompt
    if "," in default_suffix_prompt:
        # only select the last one.
        default_suffix_prompt = default_suffix_prompt.split(",")[-1].strip()
    return default_suffix_prompt

def self_infill_split(tokenizer, gen_code, fim_sentinel_dict=None):
    result_dict = {}

    splits_by_prefix = gen_code.split(fim_sentinel_dict["fim_prefix"], 1)
    if len(splits_by_prefix) == 2:
        result_dict["fim_prefix_present"] = True
        _, gen_code_without_prefix = splits_by_prefix
    else:
        result_dict["fim_prefix_present"] = False
        gen_code_without_prefix = splits_by_prefix[0]

    splits_by_suffix = gen_code_without_prefix.split(fim_sentinel_dict["fim_suffix"], 1)
    if len(splits_by_suffix) == 2:
        result_dict["fim_suffix_present"] = True
        prefix, rest = splits_by_suffix
    else:
        result_dict["fim_suffix_present"] = False
        prefix = splits_by_suffix[0]
        rest = ""

    splits_by_middle = rest.split(fim_sentinel_dict["fim_middle"], 1)
    if len(splits_by_middle) == 2:
        result_dict["fim_middle_present"] = True
        suffix, infill = splits_by_middle
    else:
        result_dict["fim_middle_present"] = False
        suffix = splits_by_middle[0]
        infill = ""

    if not isinstance(fim_sentinel_dict["fim_ending"], list):
        fim_ending_candidates = [fim_sentinel_dict["fim_ending"]]
    else:
        fim_ending_candidates = fim_sentinel_dict["fim_ending"]
    # greedily find the end according to the order in the list
    for ending in fim_ending_candidates:
        splits_by_eot = infill.split(ending, 1)
        if len(splits_by_eot) == 2:
            result_dict["fim_ending_present"] = True
            infill, _ = splits_by_eot
            break
        else:
            result_dict["fim_ending_present"] = False
            infill = splits_by_eot[0]

    if (
        result_dict["fim_prefix_present"] and
        result_dict["fim_middle_present"] and
        result_dict["fim_suffix_present"] and
        (not result_dict["fim_ending_present"])
    ):
        # sometimes the infill might not join the suffix well. 
        # In this case, just discard the suffix. 
        suffix = ""

    for k in [
        "bos_token", 
        "eos_token", 
        "unk_token", 
        "pad_token", 
        "additional_special_tokens"
    ]:
        v = tokenizer.special_tokens_map.get(k, None)
        if v:
            if k == "additional_special_tokens":
                for t in v:
                    prefix = prefix.replace(t, "")
                    infill = infill.replace(t, "")
                    suffix = suffix.replace(t, "")
            else:
                prefix = prefix.replace(v, "")
                infill = infill.replace(v, "")
                suffix = suffix.replace(v, "")
    
    result_dict["split"] = (prefix, infill, suffix)
    return result_dict

