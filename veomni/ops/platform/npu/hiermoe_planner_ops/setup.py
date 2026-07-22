import glob
import os

import torch_npu
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch_npu.utils.cpp_extension import NpuExtension


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
TORCH_NPU_DIR = os.path.dirname(os.path.abspath(torch_npu.__file__))

setup(
    name="veomni-hiermoe-npu-ops",
    version="0.1.0",
    ext_modules=[
        NpuExtension(
            name="_hiermoe_npu_ops",
            sources=glob.glob(os.path.join(BASE_DIR, "csrc", "extension", "*.cpp")),
            extra_compile_args=[
                "-O3",
                "-I" + os.path.join(TORCH_NPU_DIR, "include", "third_party", "acl", "inc"),
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
