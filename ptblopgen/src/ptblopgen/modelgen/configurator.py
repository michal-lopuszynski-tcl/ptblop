import pydantic


class TrnSchedulerEntryConfig(pydantic.BaseModel):
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
    n_val_rand: int
    trn_schedule: list[TrnSchedulerEntryConfig]


class ParetoOptimizationConfig(pydantic.BaseModel):
    n_gen: int
    pop_size: int
    optimizer_seed: int
