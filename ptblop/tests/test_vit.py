import typing

import helpers
import pytest

if typing.TYPE_CHECKING:
    import timm  # type: ignore
else:
    try:
        import timm
    except ModuleNotFoundError:
        timm = None

import torch

import ptblop


def make_vit() -> helpers.MODEL_DATA_TYPE:
    model = timm.create_model(
        "vit_tiny_patch16_224.augreg_in21k_ft_in1k", pretrained=False
    )
    bp_config0 = ptblop.get_unpruned_bp_config(model)
    generator_cpu = torch.Generator()

    def __get_vit_data() -> torch.Tensor:
        return torch.rand(1, 3, 224, 224, generator=generator_cpu)

    return model, __get_vit_data, bp_config0


@pytest.mark.skipif(timm is None, reason="timm not installed")
def test_vit_unpruned_forward_cpu() -> None:
    helpers.check_unpruned_forward(make_vit, torch.device("cpu"))


@pytest.mark.skipif(timm is None, reason="timm not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_vit_unpruned_forward_gpu() -> None:
    helpers.check_unpruned_forward(make_vit, torch.device("cuda"))


@pytest.mark.skipif(timm is None, reason="timm not installed")
def test_vit_decomposed1_cpu() -> None:
    helpers.check_disabled_attentnions(make_vit, torch.device("cpu"))


@pytest.mark.skipif(timm is None, reason="timm not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_vit_decomposed1_gpu() -> None:
    helpers.check_disabled_attentnions(make_vit, torch.device("cuda"))


@pytest.mark.skipif(timm is None, reason="timm not installed")
def test_vit_disabled_mlps_cpu() -> None:
    helpers.check_disabled_mlps(make_vit, torch.device("cpu"))


@pytest.mark.skipif(timm is None, reason="timm not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_vit_disabled_mlps_gpu() -> None:
    helpers.check_disabled_mlps(make_vit, torch.device("cuda"))


@pytest.mark.skipif(timm is None, reason="timm not installed")
def test_vit_disabled_blocks_cpu() -> None:
    helpers.check_disabled_blocks(make_vit, torch.device("cpu"))


@pytest.mark.skipif(timm is None, reason="timm not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_vit_disabled_blocks_gpu() -> None:
    helpers.check_disabled_blocks(make_vit, torch.device("cuda"))


@pytest.mark.skipif(timm is None, reason="timm not installed")
def test_vit_enable_disable_cpu() -> None:
    helpers.check_enable_disable(make_vit, torch.device("cpu"))


@pytest.mark.skipif(timm is None, reason="timm not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_vit_enable_disable_gpu() -> None:
    helpers.check_enable_disable(make_vit, torch.device("cuda"))


@pytest.mark.skipif(timm is None, reason="timm not installed")
def test_vit_num_params() -> None:
    helpers.check_num_params(make_vit)


@pytest.mark.skipif(timm is None, reason="timm not installed")
def test_vit_disabled_block0_is_identity_cpu() -> None:
    helpers.check_disabled_block_is_identity(
        make_vit, torch.device("cpu"), "blocks.0", 0
    )


@pytest.mark.skipif(timm is None, reason="timm not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_vit_disabled_block0_is_identity_gpu() -> None:
    helpers.check_disabled_block_is_identity(
        make_vit, torch.device("cuda"), "blocks.0", 0
    )


@pytest.mark.skipif(timm is None, reason="timm not installed")
def test_vit_disabled_block5_is_identity_cpu() -> None:
    helpers.check_disabled_block_is_identity(
        make_vit, torch.device("cpu"), "blocks.5", 5
    )


@pytest.mark.skipif(timm is None, reason="timm not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_vit_disabled_block5_is_identity_gpu() -> None:
    helpers.check_disabled_block_is_identity(
        make_vit, torch.device("cuda"), "blocks.5", 5
    )
