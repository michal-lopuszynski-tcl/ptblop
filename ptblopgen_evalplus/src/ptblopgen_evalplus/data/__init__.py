from .utils import load_solutions


# def get_evalperf_data():
#     import json
#     from datasets import load_dataset

#     dataset = load_dataset("evalplus/evalperf", split="test").to_list()
#     for d in dataset:
#         d["pe_input"] = json.loads(d["pe_input"])
#     return {task["task_id"]: task for task in dataset}
