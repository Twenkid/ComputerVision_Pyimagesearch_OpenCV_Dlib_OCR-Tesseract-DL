#%%cython
#cython test
#%cython -a
#%load_ext Cython

import cython

#@cython.boundscheck(False)
%%cython
def kur(int a, int b):
  cdef int i;
  i = a+b
  return i
  

  
def main(i):
  print(i)
  s = dir(cython)
  for k in s: print(k)
  return i
  
  
  
#cpdef unsigned char[:, :] threshold_fast(int a, int b):