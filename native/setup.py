"""rerank_cpp 네이티브 확장 빌드 스크립트.

실행 (native/ 디렉터리에서):
    python setup.py build_ext --inplace
"""
import pybind11
from setuptools import Extension, setup

ext_modules = [
    Extension(
        "rerank_cpp",
        ["rerank_cpp.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["/O2", "/std:c++17", "/utf-8", "/openmp"],
    ),
]

setup(
    name="rerank_cpp",
    version="0.1.0",
    ext_modules=ext_modules,
)
