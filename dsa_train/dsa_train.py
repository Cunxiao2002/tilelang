import torch
import tilelang
import tilelang.language as T
from kernels import sparse_gqa_bwd_interface, sparse_gqa_fwd_interface, ref_sparse_gqa_fwd_interface, ref_sparse_gqa_bwd_interface
from kernels import assert_tensors_similar
import argparse


class _DSAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, indices, sm_scale=None, return_p_sum=False, d_v=64, block_I=64, num_stages=2, threads=256, is_causal=True):
        o, lse = sparse_gqa_fwd_interface(
            q,
            k,
            v,
            indices,
            sm_scale=sm_scale,
            return_p_sum=return_p_sum,
            d_v=d_v,
            block_I=block_I,
            num_stages=num_stages,
            threads=threads,
        )
        ctx.save_for_backward(q, k, v, o, lse, indices)
        ctx.causal = is_causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, indices = ctx.saved_tensors        
        def maybe_contiguous(x):
            if x.stride(-1) != 1:
                x = x.contiguous()
            return x
        
        do, q, k, v, o = [maybe_contiguous(x) for x in [do, q, k, v, o]]
        
        dq, dk, dv = sparse_gqa_bwd_interface(q,
                                              k,
                                              v,
                                              o,
                                              do,
                                              indices,
                                              lse,
                                              is_casual=True)
        return dq, dk, dv, None, None, None, None, None, None, None, None

DSAttention = _DSAttention.apply
def main(B: int,
         S: int,
         SKV: int,
         H: int,
         HKV: int,
         DQK: int,
         DV: int,
         topk: int = 2048,
         check_correctness: bool = False,
         is_causal=True):
    flops_per_qk = 2.0 * B * H * S * S * DQK
    flops_per_v = 2.0 * B * H * S * S * DV
    total_flops = 3 * flops_per_qk + 2 * flops_per_v
    dtype=torch.bfloat16
    if is_causal:
        total_flops *= 0.5
    
    q = torch.randn((B, S, H, DQK), dtype=dtype, device='cuda').requires_grad_(True)
    k = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device='cuda').requires_grad_(True)
    v = torch.randn((B, SKV, HKV, DV), dtype=dtype, device='cuda').requires_grad_(True)
    do = torch.randn((B, S, H, DV), dtype=dtype, device='cuda')

    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                indices[b, t, h, :len(i_i)] = i_i
    O = DSAttention(q, k, v, indices)
    O.backward(do, retain_graph=True)
    dq, q.grad = q.grad.clone(), None
    dk, k.grad = k.grad.clone(), None
    dv, v.grad = v.grad.clone(), None


    if check_correctness:
        O_ref = ref_sparse_gqa_fwd_interface(q, k, v, indices)
        O_ref.backward(do, retain_graph=True)
        dq_ref, q.grad = q.grad.clone(), None
        dk_ref, k.grad = k.grad.clone(), None
        dv_ref, v.grad = v.grad.clone(), None

        assert_tensors_similar(O, O_ref, eps=1e-4)
        assert_tensors_similar(dq, dq_ref, eps=1e-4)
        assert_tensors_similar(dk, dk_ref, eps=1e-4)
        assert_tensors_similar(dv, dv_ref, eps=1e-4)
        print("All check pass")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--B', type=int, default=1, help='Batch size')
    parser.add_argument('--S', type=int, default=4096, help='q sequence length')
    parser.add_argument('--SKV', type=int, default=4096, help='kv sequence length')
    parser.add_argument('--H', type=int, default=32, help='Head nums for Q')
    parser.add_argument('--HKV', type=int, default=2, help='Head nums for K/V')
    parser.add_argument('--DQK', type=int, default=64, help='Head dimension for Q/K')
    parser.add_argument('--DV', type=int, default=64, help='Head dimension for V')
    parser.add_argument('--topk', type=int, default=2048, help='topk')
    parser.add_argument('--check_correctness', action='store_true', help='Check correctness')
    parser.add_argument('--is_causal', action='store_true', help='Causal flag')


    args = parser.parse_args()
    B = args.B
    S = args.S
    SKV = args.SKV
    H = args.H
    HKV = args.HKV
    DQK = args.DQK
    DV = args.DV
    topk = args.topk
    check_correctness = args.check_correctness
    is_causal = args.is_causal

    main(B, S, SKV, H, HKV, DQK, DV, topk, check_correctness, is_causal)