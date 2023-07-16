from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "Cython_Functions",
        ["Cython_Functions.pyx"],
        extra_compile_args=["-fopenmp","-O3","-ffast-math"],
        extra_link_args=["-fopenmp"]
    )
]

setup(
    name="Cython_Functions",
    ext_modules=cythonize(ext_modules)
)