import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import random

def get_ud2(m,C,d):
    ci = min(C, key = lambda c: d(m,c))
    return d(m,ci)**2

def f(M,C,d):
    sum = 0
    for m in M:
        sum += get_ud2(m,C,d)
    return sum

def distance(x,y):
    return abs(x-y)

def rbool():
    return random.choice([True, False])

def get_vector(x, shape, i):
  d = shape[1]
  return x[i*d : (i+1)*d, :].reshape(d)
def sign(n):
    return 1 if n>=0 else -1 
def l1(x, shape, M):
  (k, d) = shape

  f = 0
  g = np.zeros(x.shape)

  for i,m in enumerate(M):
    dist_all = [np.linalg.norm(m - get_vector(x, shape, j), 1) for j in range(k)] #, 1
    j = np.argmin(dist_all)
    dist_min = dist_all[j]
    diff = m - get_vector(x, shape, j)

    f += dist_min

    for l in range(d):
      g[j*d + l][0] -= sign(diff[l])

  return (f, g)

def linf(x, shape, M):
  (k, d) = shape

  f = 0
  g = np.zeros(x.shape)

  for i,m in enumerate(M):
    dist_all = [np.linalg.norm(m - get_vector(x, shape, j), np.inf) for j in range(k)] #, np.inf
    j = np.argmin(dist_all)
    diff = m - get_vector(x, shape, j)

    f += dist_all[j]

    l = np.argmax(diff)
    g[j*d + l][0] -= sign(diff[l])

  return (f, g)

smoothness = 25
quantity = 10
r = 2
c1 = 2
c2 = 7 
stop = 10

C1 = np.linspace(0, stop, smoothness)
C2 = np.linspace(0, stop, smoothness)

X, Y = np.meshgrid(C1, C2)

#M = [1, 1.2, 1.3, 2, 5, 6, 8.2, 7.3]

# TO REMEMBER
# stop 10, shift 2, range 2
# stop 10, shift 3, range 2
# stop 20, shift 3, range 5

# M = np.array([
#     c1 if rbool() else c2 for i in range(quantity)
# ]) + r * np.random.rand(quantity)

M = [3.59028018, 2.82152028, 7.34604142, 8.53012376, 3.36255176, 3.49723864,
 8.01925414, 8.44888615, 2.65308184, 3.51836114]

print(M)

plt.subplots(figsize=(10,2))
plt.scatter(M, np.zeros(quantity))
#plt.ylim([-0.1, 0.1])
plt.show()

# F = [f(M, [C1[i], C2[j]], distance)
#      for j in range(len(C2)) for i in range(len(C1))]

F = [linf(np.array([[C1[i]],[C2[j]]]), (2,1), M)[0]
     for j in range(len(C2)) for i in range(len(C1))]



Z = np.array(F).reshape(len(C1),len(C2))

#print('C1',C1)
#print('C2', C2)
#print('F', Z)

# for m in M:
#     print('(',m,', 0)')

fig = plt.figure(figsize=(10, 10))
#ax = plt.axes(projection='3d')
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')


max_f = max(F)
""" x = [0, 0, 10, 10]
y = [0, 0, 10, 10]
z = [0,  max_f, max_f, 0]
vertices = [list(zip(x, y, z))]
poly = Poly3DCollection(vertices, alpha=0.3) 
ax.add_collection3d(poly) """

""" x = [10, 10, 0, 0]
y = [0, 0, 10, 10]
z = [0, max_f, max_f, 0]
vertices = [list(zip(x, y, z))]
poly = Poly3DCollection(vertices, alpha=0.3) 
ax.add_collection3d(poly) """

ax.set_xlabel('C1')
ax.set_ylabel('C2')
ax.set_zlabel('f')

fig.tight_layout()
plt.show()
