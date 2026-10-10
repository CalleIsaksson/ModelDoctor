"""Persist enough state to resume at a completed optimizer step."""

from pathlib import Path
import random


def adapter_state(model):
    from peft import get_peft_model_state_dict
    return {name: value.detach().cpu().clone()
            for name, value in get_peft_model_state_dict(model).items()}


def restore_adapter(model, state):
    from peft import set_peft_model_state_dict
    if set(state) != set(adapter_state(model)):
        raise ValueError("Checkpoint adapter parameters do not match this model.")
    result = set_peft_model_state_dict(model, state)
    if result.unexpected_keys:
        raise ValueError(f"Unexpected adapter checkpoint keys: {result.unexpected_keys}")


def to_cpu(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(item) for item in value)
    return value


def random_state():
    import torch
    return {"python": random.getstate(), "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_random_state(state):
    import torch
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("Resuming CUDA RNG state requires the same number of GPUs.")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def resolve_checkpoint(path):
    path = Path(path)
    return path / "latest.pt" if path.is_dir() else path


def save_checkpoint(path, model, optimizer, scaler, progress, signature):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"version": 1, "signature": signature, "adapter": adapter_state(model),
             "optimizer": to_cpu(optimizer.state_dict()), "scaler": scaler.state_dict(),
             "rng": random_state(), "progress": to_cpu(progress)}
    temporary = path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path, expected_signature):
    import torch
    path = resolve_checkpoint(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("version") != 1:
        raise ValueError("Unsupported training checkpoint version.")
    if state["signature"] != expected_signature:
        differences = sorted(key for key in set(state["signature"]) | set(expected_signature)
                             if state["signature"].get(key) != expected_signature.get(key))
        raise ValueError(f"Checkpoint settings or data changed: {', '.join(differences)}")
    return state
