Command to call to build the python library from C files:

swig -python saxpy.i
python setup.py build_ext --inplace

To install in a different directory:
python setup.py install --install-platlib=.




Command to call to build the python library from C++ files:

swig -c++ -python saxpy.i
python setup.py build_ext --inplace