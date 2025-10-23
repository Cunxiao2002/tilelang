import torch
import tilelang
import tilelang.language as T

@tilelang.jit
def sparse_gqa_fwd(
    heads,
):
    pass

@tilelang.jit
def sparse_gqa_bwd(
    heads,
):
    pass


class SparseGQA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, indices):
        pass

    
    @staticmethod
    def backward(ctx, grad_output):
        pass

# 创建函数接口
def sparse_gqa_interface():
    return SparseGQA.apply()



# model中使用
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
    
    def forward(self, x):
        x = self.linear(x)
        x = sparse_gqa_interface(x)
        return x
