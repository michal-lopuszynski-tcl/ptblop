from .. import utils


def make_evaluator(evaluator_config):
    evaluator_name = evaluator_config["evaluator_name"]

    evaluator_kwargs = {
        k: v for k, v in evaluator_config.items() if k != "evaluator_name"
    }
    evaluator = utils.instantiate_from_str(evaluator_name, evaluator_kwargs)
    return evaluator
