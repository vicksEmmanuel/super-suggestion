from lm_eval.tasks.mbpp import EXAMPLARS
import fnmatch


class BaseIOProcessor:
    def __init__(self, task, task_name, tokenizer):
        self.task = task
        self.task_name = task_name
        self.tokenizer = tokenizer

    def process_input(self, doc):
        return self.task.get_prompt(doc)

    def process_output(self, output, task_id):
        """
            clean up the generation and extract the appropriate snippet
            for subsequent evaluation
            @output: the full solution (include both prompt and code)
        """
        if "insertion" in self.task_name:
            gen_code = self.task.postprocess_generation(output, int(task_id))
            return gen_code
        dataset = self.task.get_dataset()
        prompt = self.process_input(dataset[task_id])
        gen_code = output[len(prompt) :]
        gen_code = self.task.postprocess_generation(gen_code, int(task_id))
        return gen_code

    def trim_output(self, output, task_id):
        """
            remove any code beyond the current completion scope
            @output: the full solution (include both prompt and code)
        """
        dataset = self.task.get_dataset()
        prompt = self.process_input(dataset[task_id])
        gen_code = output[len(prompt) :]
        gen_code = self.task.trim_generation(gen_code, int(task_id))
        return prompt + gen_code

class StarCoderIOProcessor(BaseIOProcessor):
    def process_input(self, doc):
        """Builds the prompt for the LM to generate from."""

        ###################################
        # set up task attributes
        # for codellama and deepseek, do not strip() since 
        # \n is encoded into a single token,
        # but for starcoder models, we need to strip prompt, as
        # \n and 4 indentations are encoded into a single token
        # if (
        #     (
        #         "starcoder" in self.tokenizer.name_or_path or
        #         "santacoder" in self.tokenizer.name_or_path or
        #         "WizardCoder" in self.tokenizer.name_or_path
        #     ) and 
        #     (
        #         "humaneval" in task_name or 
        #         "mbpp" in task_name
        #     )
        # ):
        #     task.strip_prompt = True
        # else:
        #     # DS1000 and GSM8k does not apply strip() because
        #     # otherwise the model might continue to complete 
        #     # the start indicator.
        #     task.strip_prompt = False
        if self.task_name in [
            "humaneval",
            "humaneval_plus",
            "mbpp",
            "mbpp_plus"
        ]:
            return super().process_input(doc).strip()
        else:
            return super().process_input(doc)

class CodeLlamaInstructIOProcessor(BaseIOProcessor):
    def __init__(self, task, task_name, tokenizer):
        super().__init__(task, task_name, tokenizer)
        if self.task_name in [
            "mbpp_plus"
        ]:
            self.task.stop_words.append("\n[/PYTHON]")

    def process_input(self, doc):
        prompt = super().process_input(doc)
        if self.task_name in ["humaneval", "humaneval_plus"]:
            prompt = '''[INST] Write a Python function to solve the following problem:
{} [/INST] {}'''.format(prompt.rstrip(), prompt)
            return prompt
        elif self.task_name == "mbpp":
            prompt = '''[INST] {}'''.format(prompt.rstrip())[:-len("[BEGIN]\n")] + " [/INST] \n[BEGIN]\n"
            return prompt
        elif self.task_name == "mbpp_plus":
            # codellama models seem to overfit to [PYTHON]
            prompt = '''[INST] {} [/INST] \n[PYTHON]\n'''.format(prompt.rstrip())
            return prompt
        else:
            return prompt

class WizardCoderIOProcessor(BaseIOProcessor):
    def __init__(self, task, task_name, tokenizer):
        super().__init__(task, task_name, tokenizer)
        if self.task_name in [
            "humaneval",
            "humaneval_plus",
            "mbpp",
            "mbpp_plus"
        ]:
            self.task.stop_words = ["\nclass", "\nprint", "\nif"]

    def _mbpp_prompt(self, doc):
        def _format_example(text, tests, code=None):
            prompt = "{}\nYour code should satisfy the following assertion:\n```python\n{}\n```\n".format(text.strip(), "\n".join(tests))
            if code:
                code = code.replace("\r", "").replace("\t", "    ")
                prompt += "```python\n{}\n```".format(code)
            return prompt

        """Builds the prompt for the LM to generate from."""
        examples_str = ""
        for i in range(3):
            ex = EXAMPLARS[i]
            ex_prompt = _format_example(ex['text'], ex['test_list'], ex['code'])
            examples_str += "\n" + ex_prompt + "\n"

        prompt = _format_example(doc['text'], doc['test_list'], code=None)
        prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.


### Instruction:
{}

Here are some examples:
{}

### Response:""".format(prompt, examples_str)
        return prompt

    def _dsonek_completion_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        # TODO: this prompt format does not work that well.
        # currently revert to official prompts
        prompt = doc["prompt"].strip()
        prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.


### Instruction:
{}
Complete the Python code in "...".

### Response:""".format(prompt)
        return prompt


    def _base_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        prompt = doc["prompt"].strip()
        prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.


### Instruction:
Create a Python script for this problem:
{}

### Response:""".format(prompt)
        return prompt

    def process_input(self, doc):
        if self.task_name in ["humaneval", "humaneval_plus"]:
            return self._base_prompt(doc)
        elif self.task_name == "mbpp":
            return self._mbpp_prompt(doc)
        elif self.task_name == "mbpp_plus":
            # strip surrounding ```
            return self._base_prompt(doc)
        elif fnmatch.fnmatch(self.task_name, "ds1000-*"):
            return super().process_input(doc)
        else:
            return self._base_prompt(doc)

    def _cleanup_code(self, gen_code, task_id, trim_only=False):
        """
            remove any code beyond the current completion scope
            @param trim_only: if True, only remove code after the current completion scope
        """
        # adapted from https://github.com/nlpxucan/WizardLM/blob/main/WizardCoder/src/process_humaneval.py
        prompt = self.process_input(self.task.get_dataset()[task_id])
        gen_code = gen_code[len(prompt):]
        gen_code = gen_code.split("### Response:")[-1]
        gen_code = gen_code.replace("\t", "    ")
        gen_code = gen_code.split("</s>")[0]
        if "```python" in gen_code:
            def_line = gen_code.index("```python")
            gen_code = gen_code[def_line:].strip()
            if not trim_only:
            # in trim generation we do not want to
            # remove the markdown block identifier
                gen_code = gen_code.replace("```python", "")
                try:
                    next_line = gen_code.index("```")
                    gen_code = gen_code[:next_line].strip()
                except:
                    pass
        if '__name__ == "__main__"' in gen_code:
            next_line = gen_code.index('if __name__ == "__main__":')
            gen_code = gen_code[:next_line].strip()
        if "# Example usage" in gen_code:
            next_line = gen_code.index("# Example usage")
            gen_code = gen_code[:next_line].strip()
        if gen_code.startswith("Here's"):
            gen_code = gen_code.split("\n")[1:]
            gen_code = "\n".join(gen_code)
        
        if trim_only:
            return prompt + gen_code
        else:
            return gen_code
    
    def trim_output(self, output, task_id):
        if fnmatch.fnmatch(self.task_name, "ds1000-*"):
            return super().trim_output(output, task_id)
        return self._cleanup_code(output, task_id, trim_only=True)

    def process_output(self, output, task_id):
        if fnmatch.fnmatch(self.task_name, "ds1000-*"):
            return super().process_output(output, task_id)
        return self._cleanup_code(output, task_id, trim_only=False)

class DeepseekInstructIOProcessor(BaseIOProcessor):

    def __init__(self, task, task_name, tokenizer):
        super().__init__(task, task_name, tokenizer)
        if self.task_name in [
            "humaneval",
            "humaneval_plus",
            "mbpp",
            "mbpp_plus"
        ]:
            self.task.stop_words = ["\nclass", "\nprint", "\nif"]

    def _mbpp_prompt(self, doc):
        def _format_example(text, tests, code=None):
            prompt = ">>> Problem:\n{}\n>>> Test Cases:\n{}\n\n>>> Code:\n".format(text.strip(), "\n".join(tests))
            if code:
                code = code.replace("\r", "").replace("\t", "    ")
                prompt += "```python\n{}\n```".format(code)
            return prompt

        examples_str = []
        for i in range(3):
            ex = EXAMPLARS[i]
            ex_prompt = _format_example(ex['text'], ex['test_list'], ex['code'])
            example_prompt = '- Example {}:\n{}'.format(i+1, ex_prompt)
            examples_str += [example_prompt]

        prompt = _format_example(doc['text'], doc['test_list'], code=None)
        prompt = '''You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
Please refer the given examples and generate a python function for my problem.
Examples are listed as follows:
{}

Here is my problem:
{}
### Response:
'''.format('\n\n'.join(examples_str), prompt)
        return prompt
    
    def _base_prompt(self, doc):
        prompt = doc["prompt"]
        prompt = '''
Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:
```python
{}
```
'''.strip().format(prompt.strip())
        prompt = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt }],
            tokenize=False,
            add_generation_prompt=True
        )[len("<｜begin▁of▁sentence｜>"):]
        return prompt

    def process_input(self, doc):
        if self.task_name in ["humaneval", "humaneval_plus"]:
            return self._base_prompt(doc)
        elif self.task_name == "mbpp":
            return self._mbpp_prompt(doc)
        elif self.task_name == "mbpp_plus":
            return self._base_prompt(doc)
        elif fnmatch.fnmatch(self.task_name, "ds1000-*"):
            return super().process_input(doc)
        else:
            return self._base_prompt(doc)

    def _stop_at_stop_token(self, decoded_string, stop_tokens):
        """
        Produces the prefix of decoded_string that ends at the first occurrence of
        a stop_token.
        WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
        itself.
        """
        min_stop_index = len(decoded_string)
        for stop_token in stop_tokens:
            stop_index = decoded_string.find(stop_token)
            if stop_index != -1 and stop_index < min_stop_index:
                min_stop_index = stop_index
        return decoded_string[:min_stop_index]

    def _cleanup_code(self, gen_code, task_id, trim_only=False):
        """
            remove any code beyond the current completion scope
            @param trim_only: if True, only remove code after the current completion scope
        """
        prompt = self.process_input(self.task.get_dataset()[task_id])
        code = gen_code[len(prompt):]
        if "```python" in code:
            code_start_idx = code.index("```python") + len("```python")
            if trim_only:
                prompt = prompt + code[:code_start_idx]
            code = code[code_start_idx:]
            if trim_only:
                end_idx = code.find("```") + len("```") if "```" in code else len(code)
            else:
                end_idx = code.find("```") if "```" in code else len(code)
            code = code[:end_idx].rstrip()
            
        code = self._stop_at_stop_token(code, ["\nclass", "\nif", "\n#", "\nprint"])
        if trim_only:
            return prompt + code
        else:
            return code

    def trim_output(self, output, task_id):
        if fnmatch.fnmatch(self.task_name, "ds1000-*"):
            return super().trim_output(output, task_id)
        return self._cleanup_code(output, task_id, trim_only=True)

    def process_output(self, output, task_id):
        if fnmatch.fnmatch(self.task_name, "ds1000-*"):
            return super().process_output(output, task_id)
        return self._cleanup_code(output, task_id, trim_only=False)

