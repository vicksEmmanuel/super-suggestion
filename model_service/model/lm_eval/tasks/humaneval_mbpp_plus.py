"""Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation
https://arxiv.org/abs/2305.01210

The HumanEval and MBPP plus dataset

Homepage: https://github.com/evalplus/evalplus
"""

import fnmatch
import re
import os
import json
from evalplus.data import (
    get_human_eval_plus,
    get_mbpp_plus,
    write_jsonl
)
from evalplus.evaluate import evaluate
from argparse import Namespace

from lm_eval.tasks.humaneval import GeneralHumanEval

_CITATION = """
@inproceedings{evalplus,
  title = {Is Your Code Generated by Chat{GPT} Really Correct? Rigorous Evaluation of Large Language Models for Code Generation},
  author = {Liu, Jiawei and Xia, Chunqiu Steven and Wang, Yuyao and Zhang, Lingming},
  booktitle = {Thirty-seventh Conference on Neural Information Processing Systems},
  year = {2023},
  url = {https://openreview.net/forum?id=1qvx610Cu7},
}
"""

def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {multiple-py: Task, multiple-java: Task}
    """
    return {"humaneval_plus": GeneralHumanEvalPlus, "mbpp_plus": GeneralMBPPPlus}

class GeneralHumanEvalPlus(GeneralHumanEval):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = None
    DATASET_NAME = None

    def __init__(self):
        super().__init__()
        self.dataset = list(get_human_eval_plus().values())
        self.dataset_name = "humaneval"

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        test_func = doc["test"]
        entry_point = f"check({doc['entry_point']})"
        return "\n" + test_func + "\n" + entry_point
    
    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        assert len(generations) == len(self.dataset)
        # get the directory
        samples_for_eval_path = os.path.join(
            os.path.dirname(os.getenv("RUN_STATS_SAVE_PATH", "")),
            "evalplus_samples.jsonl"
        )
        
        # delete the result file if exists
        result_path = samples_for_eval_path.replace(".jsonl", "_eval_results.json")
        if os.path.isfile(result_path):
            os.remove(result_path)

        samples_for_eval = [
            dict(
                task_id = self.dataset[idx]["task_id"],
                solution = gen
            )
            for idx, generation in enumerate(generations)
            for gen in generation
        ]
        write_jsonl(samples_for_eval_path, samples_for_eval)
        flags = Namespace(
            dataset=self.dataset_name,
            samples=samples_for_eval_path,
            base_only=False,
            parallel=int(os.getenv("HF_CODE_EVAL_NUM_PROC", "1")),
            i_just_wanna_run=False,
            test_details=False,
            min_time_limit=1,
            gt_time_limit_factor=4.0,
            mini=False,
            noextreme=False,
        )
        evaluate(flags)
        return generations

class GeneralMBPPPlus(GeneralHumanEvalPlus):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = None
    DATASET_NAME = None

    def __init__(self):
        super().__init__()
        self.stop_words = ["\nclass", "\n#", "\n@", "\nprint", "\nif", "\n```"]
        self.dataset = list(get_mbpp_plus().values())
        self.dataset_name = "mbpp"

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        test_func = doc["assertion"]
        return test_func
