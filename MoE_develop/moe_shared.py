import tilelang
import torch
import tilelang.language as T
from typing import Dict, Optional
import torch.nn as nn

@tilelang.jit(pass_configs={"tl.disable_tma_lower": True, "tl.disable_warp_specialized": True})
def moe_forward_tilelang_shared(d_hidden,
                                d_expert,
                                n_shared_experts,
                                dtype,
                                num_tokens,
                                block_token=128,
                                block_dhidden=128,
                                block_dexpert=128,
                                threads=256,
                                num_stages=1):

    assert dtype == "int8"

    scale = 1.44269504  # log2(e)

    # Parameters
    dhidden = d_hidden
    dexpert = d_expert * n_shared_experts

    # Tensors: Note that input shape is reshape to (num_tokens, dhidden)
    input_shape = (num_tokens, dhidden)
    shared_W_gate_shape = (dexpert, dhidden)
    shared_W_up_shape = (dexpert, dhidden)
    shared_W_down_shape = (dhidden, dexpert)

    accum_type = "float32"

    @T.prim_func
    def kernel_shared(
            input: T.Tensor(input_shape, dtype),  # type: ignore
            shared_W_gate: T.Tensor(shared_W_gate_shape, dtype),  # type: ignore
            shared_W_up: T.Tensor(shared_W_up_shape, dtype),  # type: ignore
            shared_W_down: T.Tensor(shared_W_down_shape, dtype),  # type: ignore
            shared_expert_weight_qscale_gate: T.Tensor((n_shared_experts, dexpert), dtype),
            shared_expert_weight_qscale_up: T.Tensor((n_shared_experts, dexpert), dtype),
            shared_expert_weight_qscale_down: T.Tensor((n_shared_experts, dhidden), dtype),
            per_token_qscale_1: T.Tensor((num_tokens * n_shared_experts), dtype),
            per_token_qscale_2: T.Tensor((num_tokens * n_shared_experts), dtype),
            up_logits: T.Tensor((num_tokens, dexpert), dtype),  # type: ignore
            output: T.Tensor(input_shape, dtype),  # type: ignore
    ):
        # Step 1: Compute gate and up logits
        with T.Kernel(
                T.ceildiv(num_tokens, block_token), T.ceildiv(dexpert, block_dexpert),
                threads=threads) as (bx, by):
            # Split the block to shared experts and routed experts
            input_shared = T.alloc_fragment((block_token, block_dhidden), dtype=dtype)
            W_gate_shared = T.alloc_shared((block_dexpert, block_dhidden), dtype=dtype)
            W_up_shared = T.alloc_shared((block_dexpert, block_dhidden), dtype=dtype)
            # Shared experts: no need to check expert_indices

            gate_logits_local = T.alloc_fragment((block_token, block_dexpert), dtype=accum_type)
            up_logits_local = T.alloc_fragment((block_token, block_dexpert), dtype=accum_type)

            T.use_swizzle(10)
            T.clear(gate_logits_local)
            T.clear(up_logits_local)

            # Parallel for gate and up matmul
            for k in T.Pipelined(T.ceildiv(dhidden, block_dhidden), num_stages=num_stages):
                T.copy(input[bx * block_token, k * block_dhidden], input_shared)
                T.copy(shared_W_gate[by * block_dexpert, k * block_dhidden], W_gate_shared)
                T.copy(shared_W_up[by * block_dexpert, k * block_dhidden], W_up_shared)
                T.gemm(input_shared, W_gate_shared, gate_logits_local, transpose_B=True)
                T.gemm(input_shared, W_up_shared, up_logits_local, transpose_B=True)

            # Fuse with SiLU and element-wise product
            for i, j in T.Parallel(block_token, block_dexpert):
                gate_logits_local[i, j] = gate_logits_local[i, j] * (
                    1.0 / (1.0 + T.exp2(-gate_logits_local[i, j] * scale)))
                up_logits_local[i, j] = up_logits_local[i, j] * gate_logits_local[i, j]

            T.copy(up_logits_local, up_logits[bx * block_token, by * block_dexpert])

        # Step 2: Compute down logits
        with T.Kernel(
                T.ceildiv(num_tokens, block_token), T.ceildiv(dhidden, block_dhidden),
                threads=threads) as (bx, by):
            up_logits_shared = T.alloc_fragment((block_token, block_dexpert), dtype=dtype)
            W_down_shared = T.alloc_shared((block_dhidden, block_dexpert), dtype=dtype)
            output_local = T.alloc_fragment((block_token, block_dhidden), dtype=accum_type)

            T.use_swizzle(10)
            T.clear(output_local)

            for k in T.Pipelined(T.ceildiv(dexpert, block_dexpert), num_stages=num_stages):
                T.copy(up_logits[bx * block_token, k * block_dexpert], up_logits_shared)
                T.copy(shared_W_down[by * block_dhidden, k * block_dexpert], W_down_shared)
                T.gemm(up_logits_shared, W_down_shared, output_local, transpose_B=True)

            T.copy(output_local, output[bx * block_token, by * block_dhidden])

    return kernel_shared



class QuantLinear(nn.Module):

    def __init__(self, input_size, output_size):
        super(QuantLinear, self).__init__()
        self.input_smooth_qscale = nn.Parameter(torch.rand([1, input_size]), requires_grad=False)
        self.weight_qscale = nn.Parameter(torch.rand([1, output_size]) * 0.001, requires_grad=False)
        self.weight = nn.Parameter(
            torch.randint(-127, 127, [output_size, input_size], dtype=torch.int8),
            requires_grad=False,
        )
        self.bias = None

    def forward(self, x):
        input_smoothed = x * self.input_smooth_qscale
        amax, _ = torch.max(torch.abs(input_smoothed), dim=1)
        amax = amax[:, None]
        input_quant = input_smoothed / amax * 127.0
        input_quant = torch.floor(input_quant.to(torch.float) + 0.5)
        input_quant = torch.clip(input_quant, -127.0, 127.0).to(torch.int8)
        output = torch.matmul(input_quant.to(torch.float32), self.weight.to(torch.float32).t())
        output = output.to(torch.int32).to(torch.float32)
        output_quant = output * self.weight_qscale * amax / 127.0

        final_output = output_quant.to(torch.float32)
        return final_output.to(x.dtype)

class ExpertTorch(nn.Module):
    def __init__(self, config: Dict, d_expert: Optional[int] = None):
        super().__init__()
        self.config = config
        self.act_fn = nn.SiLU()
        self.d_hidden: int = config["d_hidden"]
        self.d_expert: int = config["d_expert"] if d_expert is None else d_expert

        self.W_gate = QuantLinear(self.d_hidden, self.d_expert)
        self.W_up = QuantLinear(self.d_hidden, self.d_expert)
        self.W_down = QuantLinear(self.d_expert, self.d_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(self.W_gate(x))
        out = self.W_down(gate * self.W_up(x))
        return out

def main():
    num_tokens = 320
    hidden_dim = 4096
    inner_dim = 1280
    num_shared_experts = 2
    dtype = "int8"

    config = {
        "d_hidden": hidden_dim,
        "d_expert": inner_dim,
        "n_shared_experts": num_shared_experts,
    }

    # expert = ExpertTorch(config, inner_dim).to("cuda")
    shared_experts = nn.ModuleList([
        ExpertTorch(config, inner_dim).to("cuda") for _ in range(num_shared_experts)
    ])

    x = torch.randn(num_tokens, hidden_dim).to(torch.bfloat16).to("cuda")
    shared_expert_output = [expert(x) for expert in shared_experts]

    final_output = torch.stack(shared_expert_output).sum(dim=0)
    
    print(f"expert_outputs: ")
    print(final_output)


if __name__ == "__main__":
    main()