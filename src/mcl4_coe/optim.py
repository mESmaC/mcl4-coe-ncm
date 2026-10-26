"""CPUOffloadAdamW and mixed-precision working weight preparation for MCL4-COE.

Adapted from mini/src/mini_moe/optim.py:
- Keeps BF16 working weights on the GPU while storing FP32 master weights,
  first/second moments (`exp_avg`, `exp_avg_sq`), and step counters in DDR5 RAM.
- Supports both `per_parameter` and `batched` pinned-memory transfer modes.
"""

from __future__ import annotations

import math
from collections import defaultdict
from copy import deepcopy

import torch


def prepare_model(
    model: torch.nn.Module,
    device: str | torch.device = "cuda",
    *,
    bf16_working_weights: bool = True,
    preserve_master_weights: bool = True,
) -> torch.nn.Module:
    """Move working weights to `device` (in BF16 for >=2D non-router weights) and keep FP32 masters on CPU."""
    target_device = torch.device(device)
    for name, parameter in model.named_parameters():
        if (
            preserve_master_weights
            and parameter.requires_grad
            and not hasattr(parameter, "_mcl4_master_cpu")
        ):
            parameter._mcl4_master_cpu = parameter.detach().to("cpu", torch.float32).clone()
        dtype = torch.float32
        if (
            bf16_working_weights
            and target_device.type == "cuda"
            and parameter.ndim >= 2
            and not name.endswith("router.weight")
        ):
            dtype = torch.bfloat16
        parameter.data = parameter.data.to(device=target_device, dtype=dtype)
    for module in model.modules():
        for buf_name, buffer in module.named_buffers(recurse=False):
            module._buffers[buf_name] = buffer.to(target_device)
    return model


def model_state_cpu(model: torch.nn.Module, optimizer=None) -> dict[str, torch.Tensor]:
    """Snapshot CPU state, preserving unrounded FP32 master weights across stages."""
    parameters = dict(model.named_parameters(remove_duplicate=False))
    result = {}
    for name, value in model.state_dict().items():
        parameter = parameters.get(name)
        if optimizer is not None and parameter is not None:
            state = optimizer.state.get(parameter, {})
            value = state.get("master", value)
        result[name] = value.detach().cpu().clone()
    return result


class CPUOffloadAdamW(torch.optim.Optimizer):
    """AdamW optimizer maintaining FP32 master weights and moments in system RAM."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        *,
        transfer_mode: str = "batched",
    ) -> None:
        if not math.isfinite(lr) or lr < 0:
            raise ValueError("lr must be finite and nonnegative")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if len(betas) != 2 or not all(0 <= beta < 1 for beta in betas):
            raise ValueError("betas must contain two values in [0, 1)")
        if transfer_mode not in {"per_parameter", "batched"}:
            raise ValueError("transfer_mode must be 'per_parameter' or 'batched'")
        self.transfer_mode = transfer_mode
        self._transfer_buffers: dict = {}
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        for group in self.param_groups:
            for parameter in group["params"]:
                original = getattr(parameter, "_mcl4_master_cpu", None)
                master = (
                    original
                    if original is not None
                    else parameter.detach().to(device="cpu", dtype=torch.float32).clone()
                )
                self.state[parameter] = {
                    "master": master,
                    "step": 0,
                    "exp_avg": torch.zeros_like(master),
                    "exp_avg_sq": torch.zeros_like(master),
                }
                if original is not None:
                    del parameter._mcl4_master_cpu

    def invalidate_transfer_buffers(self) -> None:
        """Finish pending working-weight uploads before retiring transfer scratch."""
        for buffers in self._transfer_buffers.values():
            buffers["ready"].synchronize()
        self._transfer_buffers.clear()

    def reconfigure_parameters(self, parameter_groups: list, states: dict) -> None:
        """Install a validated structural migration without resetting Adam state."""
        if len(parameter_groups) != len(self.param_groups):
            raise ValueError("Structural migration must retain optimizer parameter groups")
        groups = [
            {**group, "params": list(parameters)}
            for group, parameters in zip(self.param_groups, parameter_groups)
        ]
        parameters = [parameter for group in groups for parameter in group["params"]]
        if len(set(parameters)) != len(parameters) or set(states) != set(parameters):
            raise ValueError("Structural migration needs exactly one state per parameter")
        if any(
            parameter.grad is not None
            for group in self.param_groups + groups
            for parameter in group["params"]
        ):
            raise ValueError("Clear gradients before changing optimizer parameters")
        for parameter in parameters:
            state = states[parameter]
            if not {"master", "step", "exp_avg", "exp_avg_sq"} <= state.keys():
                raise ValueError("Structural migration requires complete Adam state")
            if isinstance(state["step"], bool) or not isinstance(state["step"], int) or state["step"] < 0:
                raise ValueError("Structural migration Adam step must be a non-negative integer")
            for key in ("master", "exp_avg", "exp_avg_sq"):
                value = state[key]
                if (
                    not isinstance(value, torch.Tensor)
                    or value.shape != parameter.shape
                    or value.dtype != torch.float32
                    or value.device.type != "cpu"
                    or not torch.isfinite(value).all()
                ):
                    raise ValueError(f"Invalid structural migration CPU FP32 state: {key}")
        self.invalidate_transfer_buffers()
        self.param_groups = groups
        self.state = defaultdict(dict, states)

    def _buffers_for(self, device: torch.device, size: int) -> dict:
        buffers = self._transfer_buffers.get(device)
        if buffers is None or buffers["gradient_cpu"].numel() < size:
            if buffers is not None:
                buffers["ready"].synchronize()
            buffers = {
                "gradient_cpu": torch.empty(size, dtype=torch.float32, pin_memory=True),
                "weight_cpu": torch.empty(size, dtype=torch.float32, pin_memory=True),
                "gradient_device": torch.empty(size, dtype=torch.float32, device=device),
                "weight_device": torch.empty(size, dtype=torch.float32, device=device),
                "ready": torch.cuda.Event(),
            }
            self._transfer_buffers[device] = buffers
        return buffers

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self.transfer_mode == "batched":
            self._step_batched()
            return loss
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.detach().to(device="cpu", dtype=torch.float32)
                state = self.state[parameter]
                state["step"] += 1
                master, average, square_average = (
                    state["master"],
                    state["exp_avg"],
                    state["exp_avg_sq"],
                )
                master.mul_(1 - group["lr"] * group["weight_decay"])
                average.lerp_(gradient, 1 - beta1)
                square_average.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                correction1 = 1 - beta1 ** state["step"]
                correction2 = 1 - beta2 ** state["step"]
                denominator = square_average.sqrt().div_(math.sqrt(correction2)).add_(group["eps"])
                master.addcdiv_(average, denominator, value=-group["lr"] / correction1)
                parameter.copy_(master)
        return loss

    def _step_batched(self) -> None:
        by_device = defaultdict(list)
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    by_device[parameter.device].append(parameter)

        gradients = {}
        transfers = []
        for device, parameters in by_device.items():
            if device.type != "cuda":
                for parameter in parameters:
                    gradients[parameter] = parameter.grad.detach().to("cpu", torch.float32)
                continue
            size = sum(parameter.numel() for parameter in parameters)
            buffers = self._buffers_for(device, size)
            stream = torch.cuda.current_stream(device)
            stream.wait_event(buffers["ready"])
            torch.cat(
                [
                    parameter.grad.detach().to(dtype=torch.float32).reshape(-1)
                    for parameter in parameters
                ],
                out=buffers["gradient_device"][:size],
            )
            buffers["gradient_cpu"][:size].copy_(
                buffers["gradient_device"][:size], non_blocking=True
            )
            transfers.append((parameters, buffers, size, stream))

        for parameters, buffers, size, stream in transfers:
            stream.synchronize()
            offset = 0
            for parameter in parameters:
                count = parameter.numel()
                gradients[parameter] = buffers["gradient_cpu"][offset : offset + count].view(
                    parameter.shape
                )
                offset += count

        for group in self.param_groups:
            parameters = [p for p in group["params"] if p.grad is not None]
            if not parameters:
                continue
            states = []
            for parameter in parameters:
                state = self.state[parameter]
                state["step"] += 1
                states.append(state)
            beta1, beta2 = group["betas"]
            masters = [s["master"] for s in states]
            averages = [s["exp_avg"] for s in states]
            squares = [s["exp_avg_sq"] for s in states]
            group_grads = [gradients[p] for p in parameters]
            torch._foreach_mul_(masters, 1 - group["lr"] * group["weight_decay"])
            torch._foreach_lerp_(averages, group_grads, 1 - beta1)
            torch._foreach_mul_(squares, beta2)
            torch._foreach_addcmul_(squares, group_grads, group_grads, value=1 - beta2)
            denominators = torch._foreach_sqrt(squares)
            torch._foreach_div_(denominators, [math.sqrt(1 - beta2 ** s["step"]) for s in states])
            torch._foreach_add_(denominators, group["eps"])
            torch._foreach_addcdiv_(
                masters,
                averages,
                denominators,
                [-group["lr"] / (1 - beta1 ** s["step"]) for s in states],
            )

        for device, parameters in by_device.items():
            if device.type != "cuda":
                by_dtype_cpu: dict[torch.dtype, tuple[list[torch.Tensor], list[torch.Tensor]]] = (
                    defaultdict(lambda: ([], []))
                )
                for p in parameters:
                    dst_l, src_l = by_dtype_cpu[p.dtype]
                    dst_l.append(p)
                    src_l.append(self.state[p]["master"])
                for dst_l, src_l in by_dtype_cpu.values():
                    torch._foreach_copy_(dst_l, src_l)
        for parameters, buffers, size, stream in transfers:
            by_dtype_gpu: dict[torch.dtype, tuple[list[torch.Tensor], list[torch.Tensor]]] = (
                defaultdict(lambda: ([], []))
            )
            offset = 0
            for parameter in parameters:
                count = parameter.numel()
                buffers["weight_cpu"][offset : offset + count].copy_(
                    self.state[parameter]["master"].reshape(-1)
                )
                view = buffers["weight_device"][offset : offset + count].view(parameter.shape)
                dst_l, src_l = by_dtype_gpu[parameter.dtype]
                dst_l.append(parameter)
                src_l.append(view)
                offset += count
            buffers["weight_device"][:size].copy_(buffers["weight_cpu"][:size], non_blocking=True)
            for dst_l, src_l in by_dtype_gpu.values():
                torch._foreach_copy_(dst_l, src_l)
            buffers["ready"].record(stream)
