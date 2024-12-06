import pydantic


class TrnSchedulerEntryConfig(pydantic.BaseModel):
    n_trn_onel: int
    n_trn_rand: int
    n_trn_actl: int
    n_trn_parf: int


class SamplerConfig(pydantic.BaseModel):
    random_bp_config_rng_seed: int
    evaluator_target: str
    max_num_changes_factor: float
    num_ranked_candidates: int
    max_random_config_trials: int
    n_val_rand: int
    trn_schedule: list[TrnSchedulerEntryConfig]
