from numba import cuda
import numpy as np

@cuda.jit
def mul(a,b,c):
  row=cuda.threadIdx.x
  col=cuda.threadIdx.y

  if row<c.shape[0] and col <c.shape[1]:
    sum=0
    for k in range(a.shape[1]):
      sum+=a[row][k]*b[k][col]

    c[row][col]=sum


n=int(input("enter size of matrix:"))

print("enter first matrics:")

a=[]

for i in range(n):
  row=list(map(int,input().split()))
  a.append(row)


print("enter second metrics:")

b=[]

for i in range(n):
  row=list(map(int,(input().split())))
  b.append(row)

a=np.array(a,dtype=np.int32)
b=np.array(b,dtype=np.int32)

c=np.zeros((n,n),dtype=np.int32)


mul[(1,1),(n,n)](a,b,c)

print("result :")
for i in range(n):
  for j in range(n):
    print(c[i][j], end=" ")
  print()