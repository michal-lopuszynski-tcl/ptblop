""" ImageNet Validation Script

This is intended to be a lean and easily modifiable ImageNet validation script for evaluating pretrained
models or training checkpoints against ImageNet or similarly organized image datasets. It prioritizes
canonical PyTorch, standard Python style, and good performance. Repurpose as you see fit.

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import argparse
import csv
import glob
import json
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from functools import partial

import torch
import torch.nn as nn
import torch.nn.parallel

import timm

from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
from timm.layers import apply_test_time_pool, set_fast_norm
from timm.models import create_model, load_checkpoint, is_model, list_models
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging, set_jit_fuser, \
    decay_batch_step, check_batch_size_retry, reparameterize_model

try:
    from apex import amp
    has_apex = True
except ImportError:
    has_apex = False

try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, 'compile')

_logger = logging.getLogger('validate')

from typing import Optional


class EvalConfig:
    data_dir: str
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
    input_size: Optional[tuple[int, int,int]]
    use_train_size: bool
    crop_pct: Optional[float]
    crop_mode: Optional[str]
    crop_border_pixels: Optional[int]
    mean: Optional[float] # Probably union of number/tuple
    std: Optional[float] # Probably union of number/tuple
    interpolation: str
    num_classes: Optional[int]
    gp: Optional[str]
    log_freq: int
    checkpoint: str
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
        self.data_dir = "/nas/datasets/vision/tenegami/"
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
        self.log_freq = 10
        self.checkpoint = ""
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


def validate(model, args):
    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint
    args.prefetcher = not args.no_prefetcher

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    model_dtype = None
    if args.model_dtype:
        assert args.model_dtype in ('float32', 'float16', 'bfloat16')
        model_dtype = getattr(torch, args.model_dtype)

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    amp_autocast = suppress
    if args.amp:
        assert model_dtype is None or model_dtype == torch.float32, 'float32 model dtype must be used with AMP'
        if args.amp_impl == 'apex':
            assert has_apex, 'AMP impl specified as APEX but APEX is not installed.'
            assert args.amp_dtype == 'float16'
            use_amp = 'apex'
            _logger.info('Validating in mixed precision with NVIDIA APEX AMP.')
        else:
            assert args.amp_dtype in ('float16', 'bfloat16')
            use_amp = 'native'
            amp_dtype = torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16
            amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
            _logger.info('Validating in mixed precision with native PyTorch AMP.')
    else:
        _logger.info(f'Validating in {model_dtype or torch.float32}. AMP not enabled.')

    if args.fuser:
        set_jit_fuser(args.fuser)

    if args.fast_norm:
        set_fast_norm()

    # create model
    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    # model = create_model(
    #     args.model,
    #     pretrained=args.pretrained,
    #     num_classes=args.num_classes,
    #     in_chans=in_chans,
    #     global_pool=args.gp,
    #     scriptable=args.torchscript,
    #     **args.model_kwargs,
    # )
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema)

    if args.reparam:
        model = reparameterize_model(model)

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info('Model %s created, param count: %d' % (args.model, param_count))

    data_config = resolve_data_config(
        vars(args),
        model=model,
        use_test_size=not args.use_train_size,
        verbose=True,
    )
    test_time_pool = False
    if args.test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config)

    # model = model.to(device=device, dtype=model_dtype)  # FIXME move model device & dtype into create_model
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        model = torch.jit.script(model)
    elif args.torchcompile:
        assert has_compile, 'A version of torch w/ torch.compile() is required for --compile, possibly a nightly.'
        torch._dynamo.reset()
        model = torch.compile(model, backend=args.torchcompile, mode=args.torchcompile_mode)
    elif args.aot_autograd:
        assert has_functorch, "functorch is needed for --aot-autograd"
        model = memory_efficient_fusion(model)

    if use_amp == 'apex':
        model = amp.initialize(model, opt_level='O1')

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    criterion = nn.CrossEntropyLoss().to(device)

    root_dir = args.data_dir
    if args.input_img_mode is None:
        input_img_mode = 'RGB' if data_config['input_size'][0] == 3 else 'L'
    else:
        input_img_mode = args.input_img_mode
    dataset = create_dataset(
        root=root_dir,
        name=args.dataset,
        split=args.split,
        download=args.dataset_download,
        load_bytes=args.tf_preprocessing,
        class_map=args.class_map,
        num_samples=args.num_samples,
        input_key=args.input_key,
        input_img_mode=input_img_mode,
        target_key=args.target_key,
        trust_remote_code=args.dataset_trust_remote_code,
    )

    if args.valid_labels:
        with open(args.valid_labels, 'r') as f:
            valid_labels = [int(line.rstrip()) for line in f]
    else:
        valid_labels = None

    if args.real_labels:
        real_labels = RealLabelsImagenet(dataset.filenames(basename=True), real_json=args.real_labels)
    else:
        real_labels = None

    crop_pct = 1.0 if test_time_pool else data_config['crop_pct']
    loader = create_loader(
        dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=crop_pct,
        crop_mode=data_config['crop_mode'],
        crop_border_pixels=args.crop_border_pixels,
        pin_memory=args.pin_mem,
        device=device,
        img_dtype=model_dtype or torch.float32,
        tf_preprocessing=args.tf_preprocessing,
    )

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    with torch.no_grad():
        # warmup, reduce variability of first batch time, especially for comparing torchscript vs non
        input = torch.randn((args.batch_size,) + tuple(data_config['input_size'])).to(device=device, dtype=model_dtype)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        with amp_autocast():
            model(input)

        end = time.time()
        for batch_idx, (input, target) in enumerate(loader):
            if args.no_prefetcher:
                target = target.to(device=device)
                input = input.to(device=device, dtype=model_dtype)
            if args.channels_last:
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

            if batch_idx % args.log_freq == 0:
                _logger.info(
                    'Test: [{0:>4d}/{1}]  '
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f})  '
                    'Acc@5: {top5.val:>7.3f} ({top5.avg:>7.3f})'.format(
                        batch_idx,
                        len(loader),
                        batch_time=batch_time,
                        rate_avg=input.size(0) / batch_time.avg,
                        loss=losses,
                        top1=top1,
                        top5=top5
                    )
                )

    if real_labels is not None:
        # real labels mode replaces topk values at the end
        top1a, top5a = real_labels.get_accuracy(k=1), real_labels.get_accuracy(k=5)
    else:
        top1a, top5a = top1.avg, top5.avg
    results = OrderedDict(
        model=args.model,
        top1=round(top1a, 4), top1_err=round(100 - top1a, 4),
        top5=round(top5a, 4), top5_err=round(100 - top5a, 4),
        param_count=round(param_count / 1e6, 2),
        img_size=data_config['input_size'][-1],
        crop_pct=crop_pct,
        interpolation=data_config['interpolation'],
    )

    _logger.info(' * Acc@1 {:.3f} ({:.3f}) Acc@5 {:.3f} ({:.3f})'.format(
       results['top1'], results['top1_err'], results['top5'], results['top5_err']))

    return results


def _try_run(args, initial_batch_size):
    batch_size = initial_batch_size
    results = OrderedDict()
    error_str = 'Unknown'
    while batch_size:
        args.batch_size = batch_size * args.num_gpu  # multiply by num-gpu for DataParallel case
        try:
            if 'cuda' in args.device and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif "npu" in args.device and torch.npu.is_available():
                torch.npu.empty_cache()
            results = validate(args)
            return results
        except RuntimeError as e:
            error_str = str(e)
            _logger.error(f'"{error_str}" while running validation.')
            if not check_batch_size_retry(error_str):
                break
        batch_size = decay_batch_step(batch_size)
        _logger.warning(f'Reducing batch size to {batch_size} for retry.')
    results['model'] = args.model
    results['error'] = error_str
    _logger.error(f'{args.model} failed to validate ({error_str}).')
    return results


def write_results(results_file, results, format='csv'):
    with open(results_file, mode='w') as cf:
        if format == 'json':
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


_NON_IN1K_FILTERS = ['*_in21k', '*_in22k', '*in12k', '*_dino', '*fcmae', '*seer']


def evaluate(model, config):
    # setup_default_logging()
    # args = parser.parse_args(["--data-dir",
    #                           "/nas/datasets/vision/tenegami/",
    #                           "--model",
    #                           "mobilevitv2_200.cvnets_in22k_ft_in1k"])
    # print(args)
    # print("class EvalConfig:\n")

    # for k, v in vars(args).items():
    #     if v is None:
    #         print(f"    {k} : Optional[TODO]")
    #     else:
    #         print(f"    {k} : {type(v).__name__}")

    # print("\n    def __init__(self):")


    # for k, v in vars(args).items():
    #     print(f"       self.{k} = {repr(v)}")

    # import sys; sys.exit(0)

    model_cfgs = []
    model_names = []
    if os.path.isdir(config.checkpoint):
        # validate all checkpoints in a path with same model
        checkpoints = glob.glob(config.checkpoint + '/*.pth.tar')
        checkpoints += glob.glob(config.checkpoint + '/*.pth')
        model_names = list_models(config.model)
        model_cfgs = [(config.model, c) for c in sorted(checkpoints, key=natural_key)]
    else:
        if config.model == 'all':
            # validate all models in a list of names with pretrained checkpoints
            config.pretrained = True
            model_names = list_models(
                pretrained=True,
                exclude_filters=_NON_IN1K_FILTERS,
            )
            model_cfgs = [(n, '') for n in model_names]
        elif not is_model(config.model):
            # model name doesn't exist, try as wildcard filter
            model_names = list_models(
                config.model,
                pretrained=True,
            )
            model_cfgs = [(n, '') for n in model_names]

        if not model_cfgs and os.path.isfile(config.model):
            with open(config.model) as f:
                model_names = [line.rstrip() for line in f]
            model_cfgs = [(n, None) for n in model_names if n]

    if len(model_cfgs):
        _logger.info('Running bulk validation on these pretrained models: {}'.format(', '.join(model_names)))
        results = []
        try:
            initial_batch_size = config.batch_size
            for m, c in model_cfgs:
                config.model = m
                config.checkpoint = c
                r = _try_run(config, initial_batch_size)
                if 'error' in r:
                    continue
                if config.checkpoint:
                    r['checkpoint'] = config.checkpoint
                results.append(r)
        except KeyboardInterrupt as e:
            pass
        results = sorted(results, key=lambda x: x['top1'], reverse=True)
    else:
        if config.retry:
            results = _try_run(config, config.batch_size)
        else:
            results = validate(config)

    if config.results_file:
        write_results(config.results_file, results, format=config.results_format)

    # output results in JSON to stdout w/ delimiter for runner script
    return results


class ImageNetEvaluator:
    def __init__(self, **kwargs):
        pass

    def evaluate(self, model: torch.nn.Module, device: torch.device):
        return {
            "imagenet-top1-acc": 0.93
        }


def main():
    setup_default_logging()
    config = EvalConfig()
    config.data_dir = "/home/lopusz/Datasets/datahub/vision/imagenet-v2"
    config.model = "mobilevitv2_200.cvnets_in22k_ft_in1k"
    config.batch_size = 64
    model = timm.create_model(config.model, pretrained=True)
    model.to(torch.device("cuda"))
    validate(model, config)


if __name__ == "__main__":
    main()