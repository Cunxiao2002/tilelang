import tilelang.testing

import example_mla_decode
from unittest import mock
import sys


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_example_mla_decode():
    with mock.patch.object(sys, 'argv', ["example_mla_decode.py"]):
        example_mla_decode.main()


if __name__ == "__main__":
    tilelang.testing.main()
