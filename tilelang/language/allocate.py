"""Memory allocation utilities for Tile-AI programs.

This module provides a set of functions for allocating different types of memory buffers
in Tile-AI programs. It wraps TVM's buffer allocation functionality with convenient
interfaces for different memory scopes.

Available allocation functions:
    - alloc_shared: Allocates shared memory buffers for inter-thread communication
    - alloc_local: Allocates local memory buffers for thread-private storage
    - alloc_fragment: Allocates fragment memory buffers for specialized operations
    - alloc_var: Allocates single-element variable buffers

Each function takes shape and dtype parameters and returns a TVM buffer object
with the appropriate memory scope.
"""

from tvm.script import tir as T


def alloc_shared(shape, dtype, scope="shared.dyn"):
    """Allocate a shared memory buffer for inter-thread communication.

    Args:
        shape (tuple): The shape of the buffer to allocate
        dtype (str): The data type of the buffer (e.g., 'float32', 'int32')
        scope (str, optional): The memory scope. Defaults to "shared.dyn"

    Returns:
        T.Buffer: A TVM buffer object allocated in shared memory
    """
    if dtype == "bool":
        # lei: This is a hack to handle bool type.
        # Because tilelang's merge smem pass cannot merge bool type currently.
        scope = "shared"
    return T.alloc_buffer(shape, dtype, scope=scope)


def alloc_local(shape, dtype, scope="local"):
    """Allocate a local memory buffer for thread-private storage.

    Args:
        shape (tuple): The shape of the buffer to allocate
        dtype (str): The data type of the buffer (e.g., 'float32', 'int32')
        scope (str, optional): The memory scope. Defaults to "local"

    Returns:
        T.Buffer: A TVM buffer object allocated in local memory
    """
    return T.alloc_buffer(shape, dtype, scope=scope)


def alloc_fragment(shape, dtype, scope="local.fragment"):
    """Allocate a fragment memory buffer for specialized operations.

    Args:
        shape (tuple): The shape of the buffer to allocate
        dtype (str): The data type of the buffer (e.g., 'float32', 'int32')
        scope (str, optional): The memory scope. Defaults to "local.fragment"

    Returns:
        T.Buffer: A TVM buffer object allocated in fragment memory
    """
    return T.alloc_buffer(shape, dtype, scope=scope)


def alloc_var(dtype, scope="local.var"):
    """Allocate a single-element variable buffer.

    Args:
        dtype (str): The data type of the buffer (e.g., 'float32', 'int32')
        scope (str, optional): The memory scope. Defaults to "local.var"

    Returns:
        T.Buffer: A TVM buffer object allocated as a single-element variable
    """
    return T.alloc_buffer([1], dtype, scope=scope)


def alloc_barrier(arrive_count: int):
    """Allocate a barrier buffer.

    Args:
        arrive_count (int): The number of threads that need to arrive at the barrier

    Returns:
        T.Buffer: A TVM buffer object allocated as a barrier
    """
    return T.alloc_buffer([arrive_count], "uint64", scope="shared.barrier")
