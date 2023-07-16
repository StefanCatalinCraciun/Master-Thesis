%module wrapper
%{
  #define SWIG_FILE_WITH_INIT
  #include "wrapper.h"
%}

%include "numpy.i"
%init %{
import_array();
%}


%apply (float* INPLACE_ARRAY1, int DIM1) {(float* x, int sizex),(float* y, int sizey)}

%include "wrapper.h"
%clear (float* INPLACE_ARRAY1, int DIM1);
