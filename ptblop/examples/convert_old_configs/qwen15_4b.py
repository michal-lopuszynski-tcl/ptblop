import json
import pathlib


def convert_qwen15_4b_old_config_to_bp_config(old_config):
    attention = old_config["attention_indices"]
    mlp = old_config["mlp_indices"]

    assert len(mlp) == 40
    assert len(attention) == 40

    res = {}
    key_tempalte = "model.layers.{0:d}"

    for i, (use_attention_int, use_mlp_int) in enumerate(zip(attention, mlp)):
        key = key_tempalte.format(i)
        assert use_attention_int in (0, 1)
        assert use_mlp_int in (0, 1)
        res[key] = {
            "use_attention": bool(use_attention_int),
            "use_mlp": bool(use_mlp_int),
        }

    return res


def load_config(config_path):
    with open(config_path, "rt") as f:
        return json.load(f)


def save_config(config_path, config):
    with open(config_path, "wt") as f:
        json.dump(config, f)


def main():
    inp_dir = pathlib.Path("inp/qwen15_4b/")
    out_dir = pathlib.Path("out/qwen15_4b/")
    out_dir.mkdir(exist_ok=True, parents=True)

    for config_name in ["config_m20.json", "config_m24.json", "config_m30.json"]:
        old_config = load_config(inp_dir / config_name)
        new_config = convert_qwen15_4b_old_config_to_bp_config(old_config)
        save_config(out_dir / f"bp_{config_name}", new_config)


if __name__ == "__main__":
    main()
