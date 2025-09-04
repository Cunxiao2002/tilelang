import torch
import pytest
import sys
import os
import tilelang

# Add project root to path to allow importing from MoE_develop, which is not a package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from MoE_develop.gather.gather import scatter_reduce_sum

@pytest.mark.parametrize("num_tokens", [4096])
@pytest.mark.parametrize("topk", [4])
@pytest.mark.parametrize("hidden_dim", [1024])
@pytest.mark.parametrize("blk_tokens", [128])
@pytest.mark.parametrize("dtype_str, torch_dtype", [("float16", torch.float16)])
def test_scatter_reduce_sum_correctness(num_tokens, topk, hidden_dim, blk_tokens, dtype_str, torch_dtype):
    """
    Tests the correctness of the scatter_reduce_sum kernel by comparing its output
    with the equivalent operation from torch.scatter_reduce.
    """
    device = "cuda"

    # Create kernel
    kernel = scatter_reduce_sum(
        num_tokens=num_tokens,
        topk=topk,
        hidden_dim=hidden_dim,
        blk_tokens=blk_tokens,
        dtype=dtype_str
    )

    # Create test data
    src_size = (num_tokens * topk, hidden_dim)
    src = torch.randn(src_size, dtype=torch_dtype, device=device)
    
    # Indices for scattering. Should be in range [0, num_tokens - 1]
    idx = torch.randint(0, num_tokens, (num_tokens * topk,), dtype=torch.int32, device=device)
    
    # Output tensor for tilelang kernel, initialized to zeros
    output_tl = torch.zeros(num_tokens, hidden_dim, dtype=torch_dtype, device=device)

    # Run tilelang kernel
    kernel(src=src, idx=idx, output=output_tl)
    torch.cuda.synchronize()

    # --- Reference implementation with torch.scatter_reduce ---
    output_ref = torch.zeros(num_tokens, hidden_dim, dtype=torch_dtype, device=device)
    
    # torch.scatter_reduce expects index to have same number of dims as src and be of type LongTensor (int64).
    idx_torch = idx.to(torch.int64).unsqueeze(-1).expand_as(src)
    
    # Perform the scatter-reduce operation
    output_ref = output_ref.scatter_reduce(0, idx_torch, src, reduce="sum", include_self=True)

    # Compare results
    # The kernel in MoE_develop/gather/gather.py has a bug and is expected to fail this test.
    # This test serves to demonstrate the incorrectness.
    torch.testing.assert_close(output_tl, output_ref, rtol=1e-2, atol=1e-2)
