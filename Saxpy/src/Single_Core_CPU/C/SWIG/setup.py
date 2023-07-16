from distutils.core import setup, Extension
import numpy
import os

wrapper_module = Extension('_wrapper'
                           ,sources = ['wrapper_wrap.c', 'wrapper.c'],
                           include_dirs=[numpy.get_include(),'.'],
                           extra_compile_args=['-fopenmp', '-Ofast', '-march=native', '-mavx512f', '-ffast-math'],
                           extra_link_args=['-fopenmp', '-Ofast', '-march=native', '-mavx512f', '-ffast-math'],)

setup (name = 'simple',
       version = '1.0',
       author      = "Stefan",
       description = """C wrapper for Python""",
       ext_modules = [wrapper_module],
       py_modules = ["wrapper"],
       )