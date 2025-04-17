import typing

import pydantic


class TrnDataBlockConfig(pydantic.BaseModel):
    n_trn_zerl: int
    n_trn_onel: int
    n_trn_rand: int
    n_trn_actl: int
    n_trn_parf: int


class SamplerConfig(pydantic.BaseModel):
    random_bp_config_rng_seed: int
    quality_evaluator_metric: str
    cost_metric: str
    max_num_changes_factor: float
    actl_num_scored_candidates: int
    parf_min_quality_evaluator_metric: float
    n_data_iter: int
    n_val_rand: int
    trn_data_iter_configs: list[TrnDataBlockConfig]
    full_block_mode: bool


class PymooParetoOptimizationConfig(pydantic.BaseModel):
    optimizer: typing.Literal["pymoo"]
    n_gen: int
    pop_size: int
    optimizer_seed: int
    sampling_mode: typing.Literal["binomial", "uniform"]
    sampling_p: float


class BeamParetoOptimizationConfig(pydantic.BaseModel):
    optimizer: typing.Literal["beam"]
    beam_size: int
    pareto_size: int
    max_num_changes_factor: float
