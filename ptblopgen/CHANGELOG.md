# TODO
+ **!!.** Add option to customize cost model
+ **!!.** Add simple id to saved PFs
+ **!..** Add support for vision transformers evaluation
+ **!..** Consider moving to string bp_config signatures
+ **!..** Add num-layers as cost metric
+ **!..** Refactor the output so the metrics are in a subfield
+ **...** In LLM evaluator, add batch size handling to perplexity evaluator

# DONE

## 0.1.0
+ Add pluggable LLM evaluators
+ Add full block mode
+ Add support for vision transformers blocks
+ Add versioning zips
+ Add handling of restarts
+ Refactor all `bpconfigs` to `bp_configs`
+ In modelgen refactor `fixed_kwargs` dict
+ Rename `regressors` to `estimators`
+ Add zip target to the Makefile
+ Move generic "repro" logic from run_gen.py to run.py
