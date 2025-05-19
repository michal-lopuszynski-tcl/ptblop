import torch


class EvalplusEvaluator:
    def __init__(self, evaluator_metrics):
        self.evaluator_metrics = evaluator_metrics

    def __call__(self, model: torch.nn.Module, device: torch.device):
        return {metric: 1.0 for metric in self.evaluator_metrics}
