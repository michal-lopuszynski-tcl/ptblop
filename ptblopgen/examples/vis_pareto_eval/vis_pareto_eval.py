import pathlib
import yaml

import ptblopgen.modelgen.run_paretoeval

CONIG_PATH = "/nas/people/michal_lopuszynski/JOBS_BP5/2025-04-05_qwen25-32bCO_mbpp_pareto/config.yaml"
PARETO_EVALUATED_PATH = "/nas/people/michal_lopuszynski/JOBS_BP5/2025-04-05_qwen25-32bCO_mbpp_pareto/out/pareto_fronts/pareto_front_0000_evaluated.json"
PARETO_EVALUATED_PLOT_PATH = "test.png"

ROOT_DIR = pathlib.Path("/nas/people/michal_lopuszynski/JOBS_BP5/")
# ROOT_DIR = pathlib.Path("/mnt/cgc/tcldata/people/michal.lopuszynski/2025-04-06_qwen25-32bCO_mbpp_blocks_pareto/")
# ROOT_DIR = pathlib.Path("/mnt/cgc/tcldata/people/michal.lopuszynski/2025-04-06_qwen25-32bCO_mbpp_pareto2/")
# ROOT_DIR = pathlib.Path("/nas/people/michal_lopuszynski/JOBS_BP4/")


def make_plot(config_path, pareto_evaluated_path):
    with open(config_path, "rt") as f:
        config = yaml.safe_load(f)
    ptblopgen.modelgen.run_paretoeval.eval_pareto_front(config, pareto_evaluated_path)
    return pareto_evaluated_path.parent / (pareto_evaluated_path.stem + ".png")


def main(root_dir):
    pareto_evaluated_paths = root_dir.rglob("*evaluated.json")
    for pareto_evaluated_path in pareto_evaluated_paths:
        config_path = pareto_evaluated_path.parent.parent.parent / "config.yaml"
        config_ok = config_path.exists()
        if config_ok:
            pareto_evaluated_plot_path = make_plot(config_path, pareto_evaluated_path)
            print(f"Processing {pareto_evaluated_plot_path}")
        else:
            print(f"Skipping {pareto_evaluated_path}, confing not found")


if __name__ == "__main__":
    main(ROOT_DIR)
