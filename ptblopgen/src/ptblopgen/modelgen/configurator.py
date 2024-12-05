import pydantic


class SamplerConfig(pydantic.BaseModel):
    random_bp_config_rng_seed: int
    evaluator_target: str
    max_num_changes_factor: float
    num_ranked_candidates: int
    max_random_config_trials: int

    n_val_rand: int
    n_trn_onel_initial: int
    n_trn_rand_initial: int
    n_trn_rand_periter: int
