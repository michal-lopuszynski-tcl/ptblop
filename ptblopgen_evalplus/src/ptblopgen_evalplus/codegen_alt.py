import time


def make_prompt(task, tokenizer):
    prompt = task["prompt"].strip() + "\n"


def generate(model, tokenizer, prompt, do_sample, cfg):
    input_tokens = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    kwargs = {}
    if do_sample:
        kwargs["top_p"] = 0.95
        kwargs["temperature"] = cfg.temperature
    t_start = time.perf_counter()
    outputs = model.generate(
        input_tokens,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=do_sample,
        num_return_sequences=min(cfg.batch_size, cfg.num_samples),
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        stop_strings=cfg.eos,
        tokenizer=cfg.tokenizer,
        **kwargs,
    )
    time_gen = time.perf_counter() - t_start

    gen_strs = tokenizer.batch_decode(
        outputs[:, input_tokens.size(-1) :],
        skip_special_tokens=cfg.skip_special_tokens,
    )
    outputs_final = []
    # removes eos tokens.
    for output in gen_strs:
        min_index = 10000
        for eos in cfg.eos:
            if eos in output:
                min_index = min(min_index, output.index(eos))
        outputs_final.append(output[:min_index].replace("\t", "    "))

    outputs_raw = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    r = {
        "outputs": outputs_final,
        "prompt_raw": prompt,
        "outputs_raw": outputs_raw,
        "time_gen": time_gen,
    }
    return r


def run_codegen_task(model, tokenizer, task, greedy, enable_thinking):
    instruction_prefix = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
    response_prefix = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:"

    prompt = make_prompt(task, tokenizer)


def run_codegen(
    model,
    tokenizer,
    dataset,
    dataset_dict,
    greedy,
    enable_thinking,
):
    res = {}

    for i, (task_id, task) in enumerate(dataset_dict.items(), start=1):
        codegen_task = {"task_id": task_id} | run_codegen_task(
            model, tokenizer, task, greedy, enable_thinking
        )
        codegen_task["_identifier"] = task_id + f" (line {i+1} in memory)"

        res["task_id"] = codegen_task
    return res
