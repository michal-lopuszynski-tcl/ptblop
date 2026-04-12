import json
import logging
import time

import timm
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def setup_logging():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s.%(msecs)03d500: %(levelname).1s %(name)s.py:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Here you put modules where you want more verbose logging

    for module_name in [__name__]:
        logging.getLogger(module_name).setLevel(logging.INFO)


setup_logging()


class Timing:
    def __init__(self, block_name):
        self.block_name = block_name

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, type, value, traceback):
        duration = time.perf_counter() - self.start
        logger.info(f"{self.block_name} took {duration:.2f} sec.")


def sort_precomputed_data():
    res = []
    with open("out/results.jsonl", "rt") as f:
        for line in f:
            res.append(json.loads(line))
    res.sort(key=lambda x: x["n_params_in_million"])
    with open("results_sorted.jsonl", "wt") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    return res


def get_num_params(m: torch.nn.Module, only_trainable: bool = False) -> int:
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)


def run_scan_trasformers_only():
    """
    Checks for transformer models in timm.
    Transformer model is a model that has a submodule of type:
    `timm.models.vision_transformer.Block`.

    Outputs number of parameters, number of transformer blocks.
    """
    # model_list = ["vit_tiny_patch16_224.augreg_in21k_ft_in1k"]
    t0_start = time.perf_counter()
    model_list = timm.list_models(pretrained=True)
    res = []
    n = len(model_list)

    for i, model_name in enumerate(model_list, start=1):
        t_start = time.perf_counter()

        print(f"{i}/{n} {model_name}", end="", flush=True)
        n_transformer_blocks = 0
        n_vit_models = len(res)
        m = timm.create_model(model_name, pretrained=False)
        for submod_name, submod in m.named_modules():
            if isinstance(submod, timm.models.vision_transformer.Block):
                n_transformer_blocks += 1
        if n_transformer_blocks > 0:
            n_params_in_million = get_num_params(m) / 1.0e6
            rr = {
                "model_name": model_name,
                "n_params_in_million": n_params_in_million,
                "n_transformer_blocks": n_transformer_blocks,
                "timm": timm.__version__,
            }
            res.append(rr)
            with open("results.jsonl", "wt") as f:
                for rr in res:
                    f.write(json.dumps(rr) + "\n")
            duration = time.perf_counter() - t_start
            n_vit_models = len(res)
            print(f" ...  found BLOCKS [{n_vit_models=} {duration=:.2f} s]", flush=True)
            print(rr)
        else:
            duration = time.perf_counter() - t_start
            print(f"... no blocks,  [{n_vit_models=} {duration=:.2f} s]", flush=True)
        del m
    duration = time.perf_counter() - t0_start
    print(f"Total {duration=:.2f} s")


def run_scan_all():
    """
    List name and number of parameters for ALL timm models.
    """
    # model_list = ["vit_tiny_patch16_224.augreg_in21k_ft_in1k"]
    t0_start = time.perf_counter()

    model_list = timm.list_models(pretrained=True)
    res = []
    n = len(model_list)

    for i, model_name in enumerate(model_list, start=1):
        t_start = time.perf_counter()
        m = timm.create_model(model_name, pretrained=False)
        n_params_in_million = get_num_params(m) / 1.0e6
        rr = {
            "model_name": model_name,
            "n_params_in_million": n_params_in_million,
            "timm": timm.__version__,
        }
        res.append(rr)
        with open("results.jsonl", "wt") as f:
            for rr in res:
                f.write(json.dumps(rr) + "\n")
        duration = time.perf_counter() - t_start
        print(f"{i}/{n} {model_name} {duration=:.2f}", flush=True)
        del m
    duration = time.perf_counter() - t0_start
    print(f"Total {duration=:.2f} s")


if __name__ == "__main__":
    run_scan_trasformers_only()
    # run_scan_parameters()
