import argparse
import json
import logging

import torch
import torch._dynamo


import transformers
import ptblopgen_evalplus


logger = logging.getLogger(__name__)


def setup_logging():
    fmt = "%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(
        level=logging.WARNING,
        format=fmt,
        datefmt="%m-%d %H:%M:%S",
    )
    # Herer you put modules that you want more verbose logging

    for module_name in [__name__, "ptblopgen_evalplus"]:
        logging.getLogger(module_name).setLevel(logging.INFO)


def parse_enable_thinking(s):
    if s.lower() == "false":
        return False
    elif s.lower() == "true":
        return True
    elif s.lower() == "none":
        return None
    else:
        raise ValueError(f"{s} not in True, False, None")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model")
    parser.add_argument("--bp-config", default=None)
    parser.add_argument("--dataset")
    parser.add_argument("--limit", default=None, type=float)
    parser.add_argument("--max-new-tokens", default=None, type=int)
    parser.add_argument("--enable-thinking", default=None, type=parse_enable_thinking)

    # Alternatively: parser.parse_args(sys.argv[1:])
    return parser.parse_args()


def apply_bp_config(model, bp_config_path):
    import ptblop

    logger.info("Applying bp_config from {bp_config_path}")
    with open(bp_config_path, "rt") as f:
        bp_config = json.loads(f)
    ptblop.apply_bp_config_in_place(model, bp_config)


def main(args):
    setup_logging()
    logger.info(f"model=={args.model}")
    logger.info(f"torch=={torch.__version__}")
    logger.info(f"ptblopgen_evalplus=={ptblopgen_evalplus.__version__}")
    logger.info(f"transformers=={transformers.__version__}")
    # attn_implementation = "sdpa"
    logger.info(f"{list(transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS.keys())=}")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=torch.device("cuda"),
        # attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    if args.bp_config is not None:
        apply_bp_config(model, args.bp_config)

    model.eval()
    logger.info(f"{model.generation_config=}")
    # torch._dynamo.config.verbose = True

    # logger.info("Compiling started")
    # logger.info(f"{model.generation_config=}")
    # model.generation_config.cache_implementation = "static"
    # model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True, dynamic=True)
    # logger.info("Compiling finished")
    # torch.set_float32_matmul_precision("high")

    # torch.set_float32_matmul_precision("high")
    if args.dataset == "mbpp":
        evaluator_metrics = {"mbpp_plus": args.limit}
    elif args.dataset == "humaneval":
        evaluator_metrics = {"humaneval_plus": args.limit}
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")
    evaluator = ptblopgen_evalplus.EvalPlusEvaluator(
        tokenizer=tokenizer,
        evaluator_metrics=evaluator_metrics,
        enable_thinking=args.enable_thinking,
        max_new_tokens=args.max_new_tokens,
    )
    results_summary = evaluator(model, model.device)
    results = evaluator.get_last_results()

    results = {"model": args.model} | results

    with open("./results.json", "wt") as f:
        json.dump(results, f)
    logger.info(f"{results_summary=}")
    logger.info(f"model=={args.model}")
    logger.info(f"torch=={torch.__version__}")
    logger.info(f"ptblopgen_evalplus=={ptblopgen_evalplus.__version__}")
    logger.info(f"transformers=={transformers.__version__}")


if __name__ == "__main__":
    main(parse_args())
