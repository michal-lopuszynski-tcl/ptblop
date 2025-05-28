import argparse
import json
import logging
import pathlib

import torch
import timm
import ptblopgen_imagenet

logger = logging.getLogger(__name__)


def setup_logging():
    fmt = "%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(
        level=logging.WARNING,
        format=fmt,
        datefmt="%m-%d %H:%M:%S",
    )
    # Herer you put modules that you want more verbose logging

    for module_name in [__name__, "ptblopgen_imagenet", "timm"]:
        logging.getLogger(module_name).setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model")
    parser.add_argument("--bp-config", default=None)
    parser.add_argument("--imagenet-v1-path", default=None)
    parser.add_argument("--imagenet-v2-path", default=None)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument(
        "--results-file",
        "-r",
        default=pathlib.Path("./results.json"),
        type=pathlib.Path,
    )
    return parser.parse_args()


def apply_bp_config(model, bp_config_path):
    import ptblop

    logger.info(f"Applying bp_config from {bp_config_path}")
    with open(bp_config_path, "rt") as f:
        bp_config = json.load(f)
    ptblop.apply_bp_config_in_place(model, bp_config)


def main(args):
    setup_logging()
    logger.info(f"model=={args.model}")
    logger.info(f"torch=={torch.__version__}")
    logger.info(f"ptblopgen_imagenet=={ptblopgen_imagenet.__version__}")
    logger.info(f"timm=={timm.__version__}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = timm.create_model(args.model, pretrained=True)
    if args.bp_config is not None:
        apply_bp_config(model, args.bp_config)
    model.to(device)
    model.eval()

    evaluator_metrics = {
        "imagenet_v1_top1_acc": 1.0,
        "imagenet_v2_top1_acc": 1.0,
    }
    evaluator = ptblopgen_imagenet.ImageNetEvaluator(
        evaluator_metrics=evaluator_metrics,
        batch_size=32,
        imagenet_v1_path=args.imagenet_v1_path,
        imagenet_v2_path=args.imagenet_v2_path,
    )
    results = evaluator(model, device)

    results = {"model": args.model} | results

    with open(args.results_file, "wt") as f:
        json.dump(results, f)
    results_str = json.dumps(results, indent=2)
    logger.info(f"Results:\n\n {results_str}\n")
    logger.info(f"model=={args.model}")
    logger.info(f"torch=={torch.__version__}")
    logger.info(f"ptblopgen_imagenet=={ptblopgen_imagenet.__version__}")
    logger.info(f"timm=={timm.__version__}")


if __name__ == "__main__":
    main(parse_args())
