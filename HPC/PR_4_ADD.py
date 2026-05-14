from numba import cuda
import numpy as np


@cuda.jit
def add(a,b,c):
  i=cuda.threadIdx.x

  if i<len(c):
    c[i]=a[i]+b[i]

n=int(input("enter no of ele:"))

a=np.array(list(map(int,input("enter first vector :").split())),dtype=np.int32)
b=np.array(list(map(int,input("enter second vector :").split())),dtype=np.int32)
c=np.zeros(n,dtype=np.int32)


add[1,n](a,b,c)

print("result:")
print(c)
