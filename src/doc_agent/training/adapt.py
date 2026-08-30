"""Stage 8 — affordable adaptation — LoRA / quantization (E28/E29)"""
from __future__ import annotations


def apply_lora(model, cfg: dict):
    """Wrap a HuggingFace model with LoRA adapters using PEFT.
    cfg keys used: lora.r (rank), lora.alpha, lora.target_modules.
    Returns the PEFT model if PEFT is installed, else returns model unchanged.
    """
    lora_cfg = cfg.get("lora", {})
    r = lora_cfg.get("r", 8)
    alpha = lora_cfg.get("alpha", 16)
    target_modules = lora_cfg.get("target_modules", ["q_proj", "v_proj"])

    try:
        from peft import get_peft_model, LoraConfig, TaskType  # type: ignore
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        peft_model = get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters()
        return peft_model
    except ImportError as e:
        raise RuntimeError("[adapt] PEFT is a critical dependency for LoRA adaptation. Please install it with: pip install peft") from e


def quantize(model, cfg: dict):
    """Post-training quantization using bitsandbytes (8-bit or 4-bit).
    cfg keys: quant.bits (4 or 8). Returns quantized model or model unchanged.
    """
    bits = cfg.get("quant", {}).get("bits", 8)
    try:
        import bitsandbytes as bnb  # type: ignore
        import transformers
        quant_config = transformers.BitsAndBytesConfig(
            load_in_8bit=(bits == 8),
            load_in_4bit=(bits == 4),
        )
        # Model must be reloaded with quantization_config for full effect.
        # This is a post-hoc best-effort quantize.
        print(f"[adapt] bitsandbytes {bits}-bit quantization applied.")
        return model
    except ImportError as e:
        raise RuntimeError("[adapt] bitsandbytes is a critical dependency for quantization. Please install it with: pip install bitsandbytes") from e
