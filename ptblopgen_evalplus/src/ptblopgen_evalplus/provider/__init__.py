from .base import DecoderBase


def make_model(
    model: str,
    backend: str,
    dataset: str,
    batch_size: int = 1,
    temperature: float = 0.0,
    force_base_prompt: bool = False,
    # instruction model only
    instruction_prefix=None,
    response_prefix=None,
    # non-server only
    dtype="bfloat16",
    trust_remote_code=False,
    # vllm only
    tp=1,
    enable_prefix_caching=False,
    enable_chunked_prefill=False,
    # openai only
    base_url=None,
    # hf only
    attn_implementation="eager",
    device_map=None,
    # gptqmodel only
    gptqmodel_backend: str = "auto",
    gguf_file: str = None,
) -> DecoderBase:
    if backend == "vllm":
        from .vllm import VllmDecoder

        return VllmDecoder(
            name=model,
            batch_size=batch_size,
            temperature=temperature,
            dataset=dataset,
            force_base_prompt=force_base_prompt,
            tensor_parallel_size=tp,
            instruction_prefix=instruction_prefix,
            response_prefix=response_prefix,
            trust_remote_code=trust_remote_code,
            enable_prefix_caching=enable_prefix_caching,
            enable_chunked_prefill=enable_chunked_prefill,
            dtype=dtype,
            gguf_file=gguf_file,
        )
    elif backend == "hf":
        from .hf import HuggingFaceDecoder

        return HuggingFaceDecoder(
            name=model,
            batch_size=batch_size,
            temperature=temperature,
            dataset=dataset,
            force_base_prompt=force_base_prompt,
            instruction_prefix=instruction_prefix,
            response_prefix=response_prefix,
            attn_implementation=attn_implementation,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            gguf_file=gguf_file,
        )
    else:
        raise ValueError(f"Unknown {backend=}")
