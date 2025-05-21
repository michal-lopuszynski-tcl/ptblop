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
    parser.add_argument("--dataset")
    parser.add_argument("--limit", default=None, type=float)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--enable-thinking", default=None, type=parse_enable_thinking)

    # Alternatively: parser.parse_args(sys.argv[1:])
    return parser.parse_args()


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
    model.eval()
    logger.info(f"{model.generation_config=}")
    # torch._dynamo.config.verbose = True

    # logger.info("Compiling started")
    # logger.info(f"{model.generation_config=}")
    # model.generation_config.cache_implementation = "static"
    # model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True, dynamic=True)
    # logger.info("Compiling finished")
    # torch.set_float32_matmul_precision("high")

    torch.set_float32_matmul_precision("high")
    results = ptblopgen_evalplus.evaluate.evaluate(
        model=model,
        tokenizer=tokenizer,
        dataset=args.dataset,
        greedy=args.greedy,
        enable_thinking=args.enable_thinking,
        limit=args.limit,
    )

    results = {"model": args.model} | results

    with open("./results.json", "wt") as f:
        json.dump(results, f)
    logger.info(f"model=={args.model}")
    logger.info(f"torch=={torch.__version__}")
    logger.info(f"ptblopgen_evalplus=={ptblopgen_evalplus.__version__}")
    logger.info(f"transformers=={transformers.__version__}")


if __name__ == "__main__":
    main(parse_args())
