from tilelang import Layout, Fragment
from tvm.tir import Var

def row_major_layout():
    shape = [8, 8]
    
    def forward_fn(i, j):
        return [i * 8 + j]
    
    layout = Layout(shape, forward_fn)

    return layout

def inverse_demo():
    shape = [8, 8]
    def forward_fn(i, j):
        return [i * 8 + j]
    
    ori_layout = Layout(shape, forward_fn)
    print(ori_layout)
    inverse_layout = ori_layout.inverse()
    print(inverse_layout)

def replicate_demo():
    shape = [8, 8]
    
    def forward_thread_fn(i, j):
        return (i * 8 + j) // 2
    
    def forward_index_fn(i, j):
        return i * 8 + j
    
    fragment = Fragment(shape, forward_thread_fn=forward_thread_fn, forward_index_fn=forward_index_fn)
    print(fragment)
    replicated_fragment = fragment.replicate(2)
    print(replicated_fragment)

def repeat_demo():
    shape = [4, 4]
    def forward_thread_fn(i, j):
        # 每个thread处理连续的2个元素，一共4个thread
        return i * 2 + (j // 2)
    
    def forward_index_fn(i, j):
        # 行主序
        return i * 4 + j
    
    fragment = Fragment(
                shape=shape,
                forward_thread_fn=forward_thread_fn,
                forward_index_fn=forward_index_fn)
    print(fragment)
    
    repeat_on_thread = fragment.repeat(repeats=(2, 1), repeat_on_thread=True)
    print(repeat_on_thread)

    repeat_on_index = fragment.repeat(repeats=(2, 1), repeat_on_thread=False)
    print(repeat_on_index)

def Reshape_demo():
    shape = [8, 8]
    def forward_index_fn(i, j):
        return i * 8 + j
    
    ori_layout = Layout(shape, forward_index_fn)
    print(ori_layout)
    reshaped_layout = ori_layout.reshape(shape=(32, 8))
    print(reshaped_layout)




if __name__ == "__main__":
    # inverse_demo()
    # replicate_demo()
    # repeat_demo()
    Reshape_demo()