"""Small authorized checks for the FieldNN model implementations."""

from __future__ import annotations

import time

import torch

from models import build_model, selective_scan


def check_scan_matches_sequential() -> None:
    torch.manual_seed(1729)
    batch, length, channels, state_size = 2, 17, 3, 4
    u = torch.randn(batch, length, channels, requires_grad=True)
    delta = (0.05 + 0.2 * torch.rand(batch, length, channels)).requires_grad_()
    b_input = torch.randn(batch, length, state_size, requires_grad=True)
    a_log = torch.randn(channels, state_size).mul_(0.1).requires_grad_()

    parallel = selective_scan(u, delta, b_input, a_log)
    a = -torch.exp(a_log.float())
    hidden = torch.zeros(batch, channels, state_size)
    sequential_states = []
    for index in range(length):
        a_t = torch.exp(delta[:, index].float().unsqueeze(-1) * a.unsqueeze(0))
        b_t = (
            delta[:, index].float().unsqueeze(-1)
            * b_input[:, index].float().unsqueeze(1)
            * u[:, index].float().unsqueeze(-1)
        )
        hidden = a_t * hidden + b_t
        sequential_states.append(hidden)
    sequential = torch.stack(sequential_states, dim=1)
    torch.testing.assert_close(parallel, sequential, rtol=2e-5, atol=2e-6)

    probe = torch.randn_like(parallel)
    inputs = (u, delta, b_input, a_log)
    parallel_grads = torch.autograd.grad(
        (parallel * probe).sum(), inputs, retain_graph=True
    )
    sequential_grads = torch.autograd.grad((sequential * probe).sum(), inputs)
    for parallel_grad, sequential_grad in zip(parallel_grads, sequential_grads):
        torch.testing.assert_close(
            parallel_grad, sequential_grad, rtol=5e-5, atol=5e-6
        )
    print("scan output and gradient match the sequential recurrence")


def check_full_length_models() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"device: {torch.cuda.get_device_name(device)}")
        print(f"torch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
    else:
        print(f"device: CPU; torch: {torch.__version__}")

    for name in ("mamba", "transformer", "unet"):
        model = build_model(name).to(device).train()
        x = torch.randn(1, 4, 8192, device=device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            baseline_memory = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)

        started = time.perf_counter()
        logits = model(x)
        if logits.shape != (1, 3, 8192):
            raise AssertionError(f"{name} returned unexpected shape {tuple(logits.shape)}")
        logits.square().mean().backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        positions_per_second = 8192 / elapsed

        if device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated(device)
            extra_peak = peak_memory - baseline_memory
            print(
                f"{name}: params={parameter_count:,}; forward+backward={elapsed:.3f}s; "
                f"input_positions/s={positions_per_second:,.0f}; "
                f"peak_allocated={peak_memory / 2**20:.1f} MiB; "
                f"incremental_peak={extra_peak / 2**20:.1f} MiB"
            )
        else:
            print(
                f"{name}: params={parameter_count:,}; forward+backward={elapsed:.3f}s; "
                f"input_positions/s={positions_per_second:,.0f}"
            )

        del model, x, logits
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    check_scan_matches_sequential()
    check_full_length_models()
