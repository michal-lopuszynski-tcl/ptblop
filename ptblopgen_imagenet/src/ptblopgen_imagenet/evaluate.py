"""ImageNet Validation Script

This is intended to be a lean and easily modifiable ImageNet validation script for
evaluating pretrained models or training checkpoints against ImageNet or similarly
organized image datasets. It prioritize canonical PyTorch, standard Python style,
 and good performance. Repurpose as you see fit.

Hacked together by Ross Wightman (https://github.com/rwightman)
Adapted for ptblob by lopusz
"""

import csv
import json
import logging
import time
from collections import OrderedDict
from contextlib import suppress
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from timm.data import (
    RealLabelsImagenet,
    create_dataset,
    create_loader,
    resolve_data_config,
)
from timm.layers import apply_test_time_pool, set_fast_norm
from timm.utils import (
    AverageMeter,
    accuracy,
    check_batch_size_retry,
    decay_batch_step,
    reparameterize_model,
    set_jit_fuser,
)

try:
    from apex import amp

    has_apex = True
except ImportError:
    has_apex = False

try:
    from functorch.compile import memory_efficient_fusion

    has_functorch = True
except ImportError:
    has_functorch = False

has_compile = hasattr(torch, "compile")

logger = logging.getLogger(__name__)


class EvalConfig:
    dataset: str
    split: str
    num_samples: Optional[int]
    dataset_download: bool
    class_map: str
    input_key: Optional[str]
    input_img_mode: Optional[str]
    target_key: Optional[str]
    dataset_trust_remote_code: bool
    model: str
    pretrained: bool
    workers: int
    batch_size: int
    img_size: Optional[int]
    in_chans: Optional[int]
    input_size: Optional[tuple[int, int, int]]
    use_train_size: bool
    crop_pct: Optional[float]
    crop_mode: Optional[str]
    crop_border_pixels: Optional[int]
    mean: Optional[float]  # Probably union of number/tuple
    std: Optional[float]  # Probably union of number/tuple
    interpolation: str
    num_classes: Optional[int]
    gp: Optional[str]
    log_freq: int
    num_gpu: int
    test_pool: bool
    no_prefetcher: bool
    pin_mem: bool
    channels_last: bool
    device: str
    amp: bool
    amp_dtype: str
    amp_impl: str
    model_dtype: Optional[str]
    tf_preprocessing: bool
    use_ema: bool
    fuser: str
    fast_norm: bool
    reparam: bool
    model_kwargs: dict
    torchcompile_mode: Optional[str]
    torchscript: bool
    torchcompile: Optional[str]
    aot_autograd: bool
    results_file: str
    results_format: str
    real_labels: str
    valid_labels: str
    retry: bool

    def __init__(self):
        self.dataset = ""
        self.split = "validation"
        self.num_samples = None
        self.dataset_download = False
        self.class_map = ""
        self.input_key = None
        self.input_img_mode = None
        self.target_key = None
        self.dataset_trust_remote_code = False
        self.model = "mobilevitv2_200.cvnets_in22k_ft_in1k"
        self.pretrained = False
        self.workers = 4
        self.batch_size = 256
        self.img_size = None
        self.in_chans = None
        self.input_size = None
        self.use_train_size = False
        self.crop_pct = None
        self.crop_mode = None
        self.crop_border_pixels = None
        self.mean = None
        self.std = None
        self.interpolation = ""
        self.num_classes = None
        self.gp = None
        self.log_freq = 100
        self.num_gpu = 1
        self.test_pool = False
        self.no_prefetcher = False
        self.pin_mem = False
        self.channels_last = False
        self.device = "cuda"
        self.amp = False
        self.amp_dtype = "float16"
        self.amp_impl = "native"
        self.model_dtype = None
        self.tf_preprocessing = False
        self.use_ema = False
        self.fuser = ""
        self.fast_norm = False
        self.reparam = False
        self.model_kwargs = {}
        self.torchcompile_mode = None
        self.torchscript = False
        self.torchcompile = None
        self.aot_autograd = False
        self.results_file = ""
        self.results_format = "csv"
        self.real_labels = ""
        self.valid_labels = ""
        self.retry = False


def validate(*, model, device, data_dir, config):
    # might as well try to validate something
    config.pretrained = config.pretrained or not config.checkpoint
    config.prefetcher = not config.no_prefetcher

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    model_dtype = None
    if config.model_dtype:
        assert config.model_dtype in ("float32", "float16", "bfloat16")
        model_dtype = getattr(torch, config.model_dtype)

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    amp_autocast = suppress
    if config.amp:
        assert (
            model_dtype is None or model_dtype == torch.float32
        ), "float32 model dtype must be used with AMP"
        if config.amp_impl == "apex":
            assert has_apex, "AMP impl specified as APEX but APEX is not installed."
            assert config.amp_dtype == "float16"
            use_amp = "apex"
            logger.info("Validating in mixed precision with NVIDIA APEX AMP.")
        else:
            assert config.amp_dtype in ("float16", "bfloat16")
            use_amp = "native"
            amp_dtype = (
                torch.bfloat16 if config.amp_dtype == "bfloat16" else torch.float16
            )
            amp_autocast = partial(
                torch.autocast, device_type=device.type, dtype=amp_dtype
            )
            logger.info("Validating in mixed precision with native PyTorch AMP.")
    else:
        logger.info(f"Validating in {model_dtype or torch.float32}. AMP not enabled.")

    if config.fuser:
        set_jit_fuser(config.fuser)

    if config.fast_norm:
        set_fast_norm()

    if config.num_classes is None:
        assert hasattr(
            model, "num_classes"
        ), "Model must have `num_classes` attr if not set on cmd line/config."
        config.num_classes = model.num_classes

    if config.reparam:
        model = reparameterize_model(model)

    param_count = sum([m.numel() for m in model.parameters()]) / 1.0e6
    logger.info("Model %s created, param count: %.2f m" % (config.model, param_count))

    data_config = resolve_data_config(
        vars(config),
        model=model,
        use_test_size=not config.use_train_size,
        verbose=True,
    )
    test_time_pool = False
    if config.test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config)

    if config.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if config.torchscript:
        assert not use_amp == "apex", "Cannot use APEX AMP with torchscripted model"
        model = torch.jit.script(model)
    elif config.torchcompile:
        assert has_compile, (
            "A version of torch w/ torch.compile() is required for --compile, possibly"
            " a nightly."
        )
        torch._dynamo.reset()
        model = torch.compile(
            model, backend=config.torchcompile, mode=config.torchcompile_mode
        )
    elif config.aot_autograd:
        assert has_functorch, "functorch is needed for --aot-autograd"
        model = memory_efficient_fusion(model)

    if use_amp == "apex":
        model = amp.initialize(model, opt_level="O1")

    if config.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(config.num_gpu)))

    criterion = nn.CrossEntropyLoss().to(device)

    if config.input_img_mode is None:
        input_img_mode = "RGB" if data_config["input_size"][0] == 3 else "L"
    else:
        input_img_mode = config.input_img_mode
    dataset = create_dataset(
        root=data_dir,
        name=config.dataset,
        split=config.split,
        download=config.dataset_download,
        load_bytes=config.tf_preprocessing,
        class_map=config.class_map,
        num_samples=config.num_samples,
        input_key=config.input_key,
        input_img_mode=input_img_mode,
        target_key=config.target_key,
        trust_remote_code=config.dataset_trust_remote_code,
    )

    if config.valid_labels:
        with open(config.valid_labels, "r") as f:
            valid_labels = [int(line.rstrip()) for line in f]
    else:
        valid_labels = None

    if config.real_labels:
        real_labels = RealLabelsImagenet(
            dataset.filenames(basename=True), real_json=config.real_labels
        )
    else:
        real_labels = None

    crop_pct = 1.0 if test_time_pool else data_config["crop_pct"]
    loader = create_loader(
        dataset,
        input_size=data_config["input_size"],
        batch_size=config.batch_size,
        use_prefetcher=config.prefetcher,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=config.workers,
        crop_pct=crop_pct,
        crop_mode=data_config["crop_mode"],
        crop_border_pixels=config.crop_border_pixels,
        pin_memory=config.pin_mem,
        device=device,
        img_dtype=model_dtype or torch.float32,
        tf_preprocessing=config.tf_preprocessing,
    )

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    with torch.inference_mode():
        # warmup, reduce variability of first batch time
        # especially for comparing torchscript vs non
        input = torch.randn((config.batch_size,) + tuple(data_config["input_size"])).to(
            device=device, dtype=model_dtype
        )
        if config.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        with amp_autocast():
            model(input)

        end = time.time()
        for batch_idx, (input, target) in enumerate(loader):
            if config.no_prefetcher:
                target = target.to(device=device)
                input = input.to(device=device, dtype=model_dtype)
            if config.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            # compute output
            with amp_autocast():
                output = model(input)

                if valid_labels is not None:
                    output = output[:, valid_labels]
                loss = criterion(output, target)

            if real_labels is not None:
                real_labels.add_result(output)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output.detach(), target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % config.log_freq == 0:
                logger.info(
                    "Test: [{0:>4d}/{1}]  Time: {batch_time.val:.3f}s"
                    " ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  Loss:"
                    " {loss.val:>7.4f} ({loss.avg:>6.4f})  Acc@1: {top1.val:>7.3f}"
                    " ({top1.avg:>7.3f})"
                    " Acc@5: {top5.val:>7.3f} ({top5.avg:>7.3f})".format(
                        batch_idx,
                        len(loader),
                        batch_time=batch_time,
                        rate_avg=input.size(0) / batch_time.avg,
                        loss=losses,
                        top1=top1,
                        top5=top5,
                    )
                )

    if real_labels is not None:
        # real labels mode replaces topk values at the end
        top1a, top5a = real_labels.get_accuracy(k=1), real_labels.get_accuracy(k=5)
    else:
        top1a, top5a = top1.avg, top5.avg
    results = OrderedDict(
        model=config.model,
        top1=round(top1a, 4),
        top1_err=round(100 - top1a, 4),
        top5=round(top5a, 4),
        top5_err=round(100 - top5a, 4),
        param_count=round(param_count / 1e6, 2),
        img_size=data_config["input_size"][-1],
        crop_pct=crop_pct,
        interpolation=data_config["interpolation"],
    )

    logger.info(json.dumps(results))
    return results


def _try_run(args, initial_batch_size):
    batch_size = initial_batch_size
    results = OrderedDict()
    error_str = "Unknown"
    while batch_size:
        args.batch_size = (
            batch_size * args.num_gpu
        )  # multiply by num-gpu for DataParallel case
        try:
            if "cuda" in args.device and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif "npu" in args.device and torch.npu.is_available():
                torch.npu.empty_cache()
            results = validate(args)
            return results
        except RuntimeError as e:
            error_str = str(e)
            logger.error(f'"{error_str}" while running validation.')
            if not check_batch_size_retry(error_str):
                break
        batch_size = decay_batch_step(batch_size)
        logger.warning(f"Reducing batch size to {batch_size} for retry.")
    results["model"] = args.model
    results["error"] = error_str
    logger.error(f"{args.model} failed to validate ({error_str}).")
    return results


def write_results(results_file, results, format="csv"):
    with open(results_file, mode="w") as cf:
        if format == "json":
            json.dump(results, cf, indent=4)
        else:
            if not isinstance(results, (list, tuple)):
                results = [results]
            if not results:
                return
            dw = csv.DictWriter(cf, fieldnames=results[0].keys())
            dw.writeheader()
            for r in results:
                dw.writerow(r)
            cf.flush()


class ImageNetEvaluator:
    def __init__(
        self,
        evaluator_metrics: dict[str, float],
        batch_size: int,
        imagenet_v1_path: Optional[str],
        imagenet_v2_path: Optional[str],
    ):
        AVAILABLE_METRICS = {
            "imagenet_v1_top1_acc",
            "imagenet_v1_top5_acc",
            "imagenet_v2_top1_acc",
            "imagenet_v2_top5_acc",
        }
        unknown_metrics = set(evaluator_metrics.keys()) - AVAILABLE_METRICS
        if unknown_metrics:
            raise ValueError(f"Unknown metrics {unknown_metrics}")
        self.config = EvalConfig()
        self.config.batch_size = batch_size
        self.imagenet_v1_path = imagenet_v1_path
        self.imagenet_v2_path = imagenet_v2_path

        for k, v in evaluator_metrics.items():
            if v is not None and abs(v - 1.0) > 1.0e-3:
                logger.info(f"{abs(v - 1.0)=}")
                raise ValueError(
                    "Subset metrics not supported, "
                    f"evaluator_metrics.{k} = {v} not 1.0 or None"
                )

        imagenet_v1 = False
        imagenet_v2 = False

        for metric in evaluator_metrics:
            if metric.startswith("imagenet_v1_"):
                imagenet_v1 = True
            elif metric.startswith("imagenet_v2_"):
                imagenet_v2 = True

        if imagenet_v1:
            if self.imagenet_v1_path is None:
                msg = "ImageNet_v1 eval requested, but imagenet_v1_path is None"
                raise ValueError(msg)
        else:
            self.imagenet_v1_path = None

        if imagenet_v2:
            if self.imagenet_v2_path is None:
                msg = "ImageNet_v2 eval requested, but imagenet_v2_path is None"
                raise ValueError(msg)
        else:
            self.imagenet_v2_path = None
        if self.imagenet_v1_path is None and self.imagenet_v2_path is None:
            raise ValueError("Both ImageNet_v1 nad ImageNet_v2 disabled")

    def __call__(self, model: torch.nn.Module, device: torch.device):
        r = {}
        if self.imagenet_v1_path is not None:
            t_start = time.perf_counter()
            r_raw = validate(
                model=model,
                device=device,
                data_dir=self.imagenet_v1_path,
                config=self.config,
            )
            time_imagenet_v1_eval = time.perf_counter() - t_start
            r["imagenet_v1_top1_acc"] = r_raw["top1"] / 100.0
            r["imagenet_v1_top5_acc"] = r_raw["top5"] / 100.0
            r["imagenet_v1_img_size"] = r_raw["img_size"]
            r["imagenet_v1_crop_pct"] = r_raw["crop_pct"]
            r["imagenet_v1_interpolation"] = r_raw["interpolation"]
            r["time_imagenet_v1_eval"] = time_imagenet_v1_eval

        if self.imagenet_v2_path is not None:
            t_start = time.perf_counter()
            r_raw = validate(
                model=model,
                device=device,
                data_dir=self.imagenet_v2_path,
                config=self.config,
            )
            time_imagenet_v2_eval = time.perf_counter() - t_start
            r["imagenet_v2_top1_acc"] = r_raw["top1"] / 100.0
            r["imagenet_v2_top5_acc"] = r_raw["top5"] / 100.0
            r["imagenet_v2_img_size"] = r_raw["img_size"]
            r["imagenet_v2_crop_pct"] = r_raw["crop_pct"]
            r["imagenet_v2_interpolation"] = r_raw["interpolation"]
            r["time_imagenet_v2_eval"] = time_imagenet_v2_eval

        return r
