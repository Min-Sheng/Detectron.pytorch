# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# --------------------------------------------------------
#!/usr/bin/env python

from __future__ import print_function

import glob
import os
import numpy as np

import torch

from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import find_packages
from setuptools import setup
from setuptools import Extension

requirements = ["torch", "torchvision"]

def get_utils_extensions():
    # Obtain the numpy include directory.  This logic works across numpy versions.
    try:
        numpy_include = np.get_include()
    except AttributeError:
        numpy_include = np.get_numpy_include()
    
    ext_modules = [
        Extension(
            name='utils.cython_bbox',
            sources=['utils/cython_bbox.pyx'],
            extra_compile_args=['-Wno-cpp'],
            include_dirs=[numpy_include]
        ),
        Extension(
            name='utils.cython_nms',
            sources=['utils/cython_nms.pyx'],
            extra_compile_args=['-Wno-cpp'],
            include_dirs=[numpy_include]
        )
    ]
    return ext_modules

setup(
    name='mask_rcnn',
    ext_modules=cythonize(get_utils_extensions())
)

def get_model_utils_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "model", "csrc")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "model._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules

setup(
    name="faster_rcnn",
    version="0.1",
    description="object detection in pytorch",
    packages=find_packages(exclude=("configs", "tests",)),
    # install_requires=requirements,
    ext_modules=get_model_utils_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)