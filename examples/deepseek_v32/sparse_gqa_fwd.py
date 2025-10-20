# ruff: noqa
import torch
import tilelang
from tilelang import language as T
from utils import assert_tensors_similar


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_gqa_fwd(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=4, #每组kv对应多少个q
    sm_scale=None,
    is_causal=True,
    CP0=True,
    block_I=64,
    num_stages=2,
    threads=256,
):
    assert dim == tilelang.math.next_power_of_2(
        dim), f"haven't check padding correctness yet, dim={dim}"
    # assert tail_dim == tilelang.math.next_power_of_2(
    #     tail_dim), f"haven't check padding correctness yet, dim={tail_dim}"
    assert is_causal == True, "non-casual is not supported"
    assert (topk %
            block_I == 0), "otherwise will load some index=0 thus causing wrong kv to be loaded"
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim))**0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)

    batch = T.symbolic("batch")
    seq_len = T.symbolic("seq_len")
    seq_len_kv = T.symbolic("seq_len_kv")

    head_kv = heads // kv_group
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    lse_shape = [batch, seq_len, heads]
    indices_dtype = "int32"
    dtype = "bfloat16"
    accum_dtype = "float"

    G = kv_group
    H = head_kv
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    # if padded_H != H:
    #     assert (
    #         kv_group == 1
    #     ), "here we solve the H padding automatically, other wise you should handle Q copy and Output copy with your mask (when kv_group == 1, use g_i * padded_H:(g_i+1) * padded_H would be handled automatically)"
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim

    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64

    @T.prim_func
    def main(
            Q: T.Tensor(q_shape, dtype),  # type: ignore
            K: T.Tensor(kv_shape, dtype),  # type: ignore
            V: T.Tensor(kv_shape, dtype),  # type: ignore
            Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
            Output: T.Tensor(o_shape, dtype),  # type: ignore
            Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(
                seq_len * REPLICATE_H, batch, kv_group, threads=threads) as (
                    bx,
                    by,
                    bz,
                ):
            Q_shared = T.alloc_shared([H_per_block, D], dtype)
            K_shared = T.alloc_shared([BI, D], dtype)
            V_shared = T.alloc_shared([BI, D], dtype)
            O_shared = T.alloc_shared([H_per_block, D], dtype)
            Lse_shared = T.alloc_shared([H_per_block], accum_dtype)
            mask = T.alloc_fragment([BI], "bool")
            mask_head = T.alloc_fragment([padded_H], "bool")

            acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            S_shared = T.alloc_shared([H_per_block, BI], dtype)
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
            alpha = T.alloc_fragment([H_per_block], accum_dtype)
            m_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)

            T.fill(acc_o, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))  # avoid -inf - inf to cause nan

            b_i, g_i = by, bz
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
            q_i = s_i
            max_kv_i = q_i

            # H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
            H0 = g_i * H
            H1 = H0 + H_per_block

            # 对于GQA，如果padding_H > kv_group, 要mask掉padding部分
            for h_i in T.Parallel(padded_H):
                global_h_idx = H0 + h_i
                local_h_idx = global_h_idx - g_i * padded_H
                mask_head[h_i] = local_h_idx < H
            # 根据mask_head load Q
            for h_i, d_i in T.Parallel(padded_H, D):
                Q_shared[h_i, d_i] = T.if_then_else(mask_head[h_i], Q[b_i, s_i, H0 + h_i, d_i], T.cast(0, dtype))

            for i_i in T.Pipelined(NI, num_stages=num_stages):

                for bi_i in T.Parallel(BI):
                    mask[bi_i] = Indices[b_i, s_i, g_i, i_i * BI + bi_i] <= max_kv_i

                for bi_i, d_i in T.Parallel(BI, D):
                    K_shared[bi_i, d_i] = K[b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i,
                                              d_i]
                    V_shared[bi_i, d_i] = V[b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i,
                                              d_i]
                                        

                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_s.dtype))
                T.gemm(
                    Q_shared,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                
                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(H_per_block):
                    alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                T.reduce_sum(acc_s, sumexp_i, dim=1)  # is this a accumulate operator?
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(H_per_block, D):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                T.copy(acc_s, S_shared)
                T.gemm(S_shared, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # Rescale
            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] /= sumexp[h_i]
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale

            T.copy(acc_o, O_shared)
            T.copy(acc_o, Output[b_i, s_i, H0:H1, :])
            T.copy(sumexp, Lse_shared)
            T.copy(sumexp, Lse[b_i, s_i, H0:H1])

    return main


def sparse_gqa_fwd_interface(q,
                             k,
                             v,
                             indices,
                             sm_scale=None,
                             return_p_sum: bool = False,
                             d_v=64,
                             block_I=64,
                             num_stages=2,
                             threads=256):
    is_casual = True
    assert return_p_sum == False, "This kernel file is for fwd only"
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous and indices.is_contiguous()
    batch, seq_len, heads, dim_plus_tail_dim = q.shape
    _, seq_len_kv, kv_group, _ = k.shape

    # assert dim_plus_tail_dim == 576, "you should assign dim otherwise"
    assert dim_plus_tail_dim == 64
    dim = d_v

    assert k.shape == v.shape
    assert k.shape[-1] == dim_plus_tail_dim
    tail_dim = dim_plus_tail_dim - dim
    assert k.shape[0] == batch
    _, _, _, topk = indices.shape
    assert indices.shape == (batch, seq_len, kv_group, topk)

    kernel = sparse_gqa_fwd(
        heads,
        dim,
        tail_dim,
        topk,
        kv_group,
        sm_scale,
        is_casual,
        block_I=block_I,
        num_stages=num_stages,
        threads=threads)
    out, lse = kernel(q, k, v, indices)
    return out, lse


def ref_sparse_gqa_fwd_interface(q, k, v, indices, sm_scale=None, is_casual=True):
    q = q.float()
    k = k.float()
    v = v.float()
    indices = indices.transpose(1, 2)
    b, sq, h_q, dim_q = q.shape
    b, sk, h_kv, dim_k = k.shape
    b, sk, h_kv, dim_v = v.shape
    # h_q = h_kv * kv_group
    # kv_group = 4 代表4个q共享同一个kv
    assert h_q % h_kv == 0
    kv_groups = h_q // h_kv
    
    compressed_casual_mask = torch.arange(0, sq, dtype=torch.int32, device="cuda").view(-1, 1) >= torch.arange(0, sk, dtype=torch.int32, device="cuda").view(1, -1)
    mask = q.new_zeros(b, h_kv, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
    mask = mask[..., :-1]
    mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
    mask = mask.view(b, h_kv, 1, sq, sk)
    q = q.view(b, sq, h_kv, -1, dim_q)
    
    # 计算attention scores
    # q (b, sq, h_kv, kv_groups, dim_q)
    # k (b, sk, h_kv, dim_k)
    # v (b, sk, h_kv, dim_v)
    # score (b, h_kv, groups, sq, sk)

    score = torch.einsum("bmghd, bngd->bghmn", q, k)
    sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale
    score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
    
    # p(b, h_kv, kv_groups, sq, sk)
    p = score.softmax(dim=-1)

    # o(b, sq, h_kv, groups, dim_v)
    o = torch.einsum("bghmn, bngd->bmghd", p.type(v.dtype), v)
    o = o.reshape(b, sq, h_q, dim_v)

    return o.to(torch.bfloat16)


def test_sparse_gqa_fwd(B=1,
                        S=4096,
                        SKV=8192,
                        H=32,
                        HKV=4,
                        DQK=64,
                        DV=64,
                        topk=2048,
                        dtype=torch.bfloat16,
                        check_correctness=True,
                        block_I=64,
                        num_stages=2,
                        threads=256):
    torch.random.manual_seed(0)
    q = torch.randn((B, S, H, DQK), dtype=dtype, device="cuda").requires_grad_(True)
    k = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device="cuda").requires_grad_(True)
    v = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device="cuda").requires_grad_(True)

    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                indices[b, t, h, :len(i_i)] = i_i

    tl_out, tl_lse = sparse_gqa_fwd_interface(
        q, k, v, indices, block_I=block_I, num_stages=num_stages, threads=threads)

    if check_correctness:
        # otherwise may cause out of memory
        ref_out = ref_sparse_gqa_fwd_interface(q, k, v, indices)
        # print(f"tl_out \n{tl_out}")
        # print(f"ref_out \n{ref_out}")
        assert_tensors_similar(tl_out, ref_out, eps=1e-2, name="out")
        print("assert_tensors_similar passed")

    def fn():
        return sparse_gqa_fwd_interface(
            q, k, v, indices, block_I=block_I, num_stages=num_stages, threads=threads)

    from tilelang.profiler import do_bench

    ms = do_bench(
        fn,
        rep=100,
        warmup=250,
    )
    print(f"Average time: {ms:.3f} ms")
    print("fwd io bandwidth = ", (B * S * DQK * topk * 2) / (ms * 1e-3) / 1e12)
    print("fwd tflops = ", (B * S * (DQK + DV) * topk * 2 * H) / (ms * 1e-3) / 1e12)


if __name__ == "__main__":
    test_sparse_gqa_fwd(
        B=1,
        S=4096,
        SKV=4096,
        H=32,
        HKV=4,
        DQK=64,
        DV=64,
        topk=2048,
        dtype=torch.bfloat16,
        check_correctness=True,
        block_I=64,
        num_stages=2,
        threads=256)