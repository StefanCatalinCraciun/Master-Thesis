from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "saxpy_cython",
        ["saxpy_cython.pyx"],
        extra_compile_args=["-fopenmp","-Ofast","-march=native","-ffast-math"],
        extra_link_args=["-fopenmp"]
    )
]

setup(
    name="saxpy_cython",
    ext_modules=cythonize(ext_modules)
)
