# ptblop

Package containing builders for block-pruned transformer models in PyTorch.

## Installation

You can install `ptblop` package via `pip`:

```bash
pip install ptblop
```

## Creating a block-pruned model

To create a block-pruned model, you need a `bp_config` usually serialized in a
JSON file. A code sample for loading block pruned language model `Qwen/Qwen1.5-4B`
from `transformers` library is included below. Sample `bp_configs` for `Qwen/Qwen1.5-4B`
are [here](./examples/convert_old_configs/out/qwen15_4b).

```python
import json

import ptblop
import transformers
import torch

bp_config_path = "./bp_config.json"
model_name = "Qwen/Qwen1.5-4B"
dtype = torch.bfloat16

model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

with open(bp_config_path, "rt") as f:
        bp_config = json.load(f)

ptblop.apply_bp_config_in_place(model, bp_config)
```
