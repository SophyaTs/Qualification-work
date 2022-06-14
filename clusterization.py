import random
import numpy as np
import math
from enum import Enum, auto
import time


def measure(func, func_params):
  start = time.time()
  result = func(*func_params)
  stop = time.time()
  #print("Elapsed", stop - start, 's')
  return result

def get_vector(x, shape, i):
  d = shape[1]
  return x[i*d: (i+1)*d, :].reshape(d)

# assume that all coordintaes of all centroids are in one vector


def u_v(m, x, shape):
  k = shape[0]
  dist = [np.linalg.norm(m - get_vector(x, shape, i)) for i in range(k)]
  i_min = np.argmin(dist)
  return (dist[i_min], i_min)

# sum of distances to the closest center


def obj_function_v(x, M, closest_centroid, shape):
    f = 0
    for j, m in enumerate(M):
        di, i = u_v(m, x, shape)
        closest_centroid[j] = i
        f += di**2
    return f

# sum of -2*(mi - xj), where mi - i-th item in dataset, xj - closest center to mi


def subgr_v(x, M, closest_centroid, shape):
    g = np.zeros(shape)
    for i, mi in enumerate(M):
        j = closest_centroid[i]
        g[j] -= 2*(mi - get_vector(x, shape, j))
    return g.reshape(x.shape)


def eucl_square(x, shape, M):
  closest_centroid = [0 for m in M]
  f = obj_function_v(x, M, closest_centroid, shape)
  g = subgr_v(x, M, closest_centroid, shape)
  #print(f)
  return (f, g)

R1 = 1e10
eps = 1e-2

def init_x(x0, M):
  n = M.shape[0]
  (k, d) = x0.shape
  r = np.zeros((n*k + x0.size, 1))

  def u(m, x):
    dist = [np.linalg.norm(m-xi) for xi in x]
    return np.argmin(dist)

  for i, m in enumerate(M):
    j = u(m, x0)
    r[i*k + j] = 1

  for i in range(k):
    for j in range(d):
      r[n*k + i*d + j] = x0[i][j]
  #print(r)
  return r


def get_centroid(x0, dimensions, i):
  (n, k, d) = dimensions
  return x0[n*k + i*d: n*k + (i+1)*d, :].reshape(d)


def eucl_square_penalty(x0, shape, M):
  f = 0
  g = np.zeros(x0.shape)
  (k, d) = shape
  n = M.shape[0]

  for i, m in enumerate(M):
    u_sum = sum(abs(x0[i*k: (i+1)*k, :]))
    #print('\n',x0[i*k : (i+1)*k, :], '   the sum of abs is', u_sum)
    if u_sum - 1 > eps:
      #print('+ R1 penalty to f', R1 * abs(u_sum - 1))
      f += R1 * abs(u_sum - 1)
    for j in range(k):
      centroid = get_centroid(x0, (n, k, d), j)
      #print(x0[n*k:,:])
      #print(centroid)
      uij = x0[i*k + j][0]
      #print('uij =',uij)
      diff = m - centroid
      dist = np.linalg.norm(diff)
      f += abs(uij) * dist**2
      '''if uij < 0:
        #print('+ R2 penalty to function', -R2 * uij)
        f -= R2 * uij # same as += R* abs(uij)'''

      def sign(n):
        return 1 if n >= 0 else -1
      gu = sign(uij)*dist**2 + sign(uij * (u_sum - 1)) * R1
      g[i*k + j][0] = gu

      gd = -2*abs(uij)*diff
      #print(gd)
      for l in range(d):
        g[n*k + j*d + l][0] += gd[l]
      #print(g)

  #print('f =',f)
  #print('g =',g)
  return (f, g)


def sign(n):
    return 1 if n >= 0 else -1


def l1(x, shape, M):
  (k, d) = shape

  f = 0
  g = np.zeros(x.shape)

  for i, m in enumerate(M):
    dist_all = [np.linalg.norm(m - get_vector(x, shape, j), 1)
                for j in range(k)]  # , 1
    j = np.argmin(dist_all)
    dist_min = dist_all[j]
    diff = m - get_vector(x, shape, j)

    f += dist_min

    for l in range(d):
      g[j*d + l][0] -= sign(diff[l])

  return (f, g)


def ellipsoid_method(function, function_params, x0, r0, eps):
    # space transformation coefficient
    dimensionality = x0.size
    beta = math.sqrt((dimensionality - 1) / (dimensionality + 1)) - 1

    # space transformation matrix
    B = np.identity(dimensionality)

    # inital radius
    r = r0

    # utility variables for storing result
    x = x0.copy()
    x_optimal = x
    f_optimal = math.inf

    itr = 0

    while True:
        itr += 1

        f, g = function(x, *function_params)
        if f < f_optimal:
            f_optimal = f
            x_optimal = x.copy()

        #print('it =',itr,'   f =',f)

        ksi = B.T.dot(g)
        norm = np.linalg.norm(ksi)

        if r * norm < eps:
            return x_optimal

        ksi /= norm
        hk = (1/(dimensionality + 1)) * r
        x -= hk * B.dot(ksi)
        B += beta * B.dot(ksi).dot(ksi.T)
        r *= dimensionality / math.sqrt(dimensionality**2 - 1)


def r_algorithm(
    function,
    function_params,
    x0,
    h0,
    q1,
    q2,
    alpha,
    epsx=0.1,
    epsg=0.1,
    nh=3,
):
  hs = h0
  B = np.identity(x0.size)
  x = x0.copy()
  xr = x0.copy()

  fr, g0 = function(x, *function_params)

  if np.linalg.norm(g0) < epsg:
    return xr

  itr = 0

  while True:
    g1 = B.T.dot(g0)

    # change of x
    dx = B.dot(g1) / np.linalg.norm(g1)

    # приріст функції??
    d = 1
    ls = 0

    # norm of x change
    ddx = 0

    while d > 0:
      itr += 1
      #if itr > 100:
      #return xr

      x -= hs * dx  # .reshape(x0.shape)
      ddx += hs * np.linalg.norm(dx)
      f1, g1 = function(x, *function_params)
      #print("\ng0 =",g0)
      #print("g1 =",g1)

      if f1 < fr:
        fr = f1
        xr = x.copy()

      if np.linalg.norm(g1) < epsg:
        return xr

      ls += 1
      if ls % nh == 0:
        hs *= q2
      if ls > 500:
        return xr

      d = dx.T.dot(g1)
      #print(d,'\n')

    itr += 1

    if ls == 1:
      hs *= q1

    if ddx < epsx:
      return xr

    dg = B.T.dot(g1-g0)
    xi = dg / np.linalg.norm(dg)
    B += (1 / alpha - 1) * B.dot(xi).dot(xi.T)
    g0 = g1


def kmeans_pp_centroids(M, k):
    '''
    initialized the centroids for K-means++
    inputs:
        M - numpy array of data points having shape (200, 2)
        k - number of clusters
    '''
    ## initialize the centroids list and add
    ## a randomly selected data point to the list
    centroids = []
    centroids.append(M[np.random.randint(M.shape[0]), :])

    ## compute remaining k - 1 centroids
    for c_id in range(k - 1):

        ## initialize a list to store distances of data
        ## points from nearest centroid
        dist = []
        for i in range(M.shape[0]):
            point = M[i]
            d = math.inf

            ## compute distance of 'point' from each of the previously
            ## selected centroid and store the minimum distance
            for c in centroids:
                temp_dist = np.linalg.norm(point - c)
                d = min(d, temp_dist)
            dist.append(d)

        ## select data point with maximum distance as our next centroid
        dist = np.array(dist)
        next_centroid = M[np.argmax(dist)]
        centroids.append(next_centroid)
        dist = []
    return np.array(centroids)


def get_random(M, k):
  return np.array([[random.uniform(M[:, j].min(), M[:,j].max()) for j in range(M.shape[1])] for i in range(k)])


class Method(Enum):
  ellipsoid = auto()
  ralg = auto()


class ObjFunction(Enum):
  eu = auto()
  eup = auto()
  l1 = auto()
  linf = auto()


def cluster(M, x, method=Method.ralg, objfunc=ObjFunction.eu, e = 0.1):
  (n, d) = M.shape
  k = x.shape[0]

  f = None
  f_params = [x.shape, M]
  x_reshaped = None
  if objfunc == ObjFunction.eup:
    f = eucl_square_penalty
    x_reshaped = init_x(x, M)
  else:
    x_reshaped = x.reshape((x.size, 1))
    if objfunc == ObjFunction.eu:
      f = eucl_square
    elif objfunc == ObjFunction.l1:
      f = l1
    """ else:
      f = linf """

  x_min = []
  if method == Method.ellipsoid:
    r0 = max([M[:, j].max() - M[:, j].min() for j in range(M.shape[1])]) / 2

    x_min = measure(
        func=ellipsoid_method,
        func_params=[
            f,
            f_params,
            x_reshaped,
            r0,
            e
        ]
    )
  else:
    x_min = measure(
        func=r_algorithm,
        func_params=[
            f,
            f_params,
            x_reshaped,
            0.99,  # q1
            1.2,  # q2
            4,  # alpha
            2,  # h0
        ]
    )

  if objfunc == ObjFunction.eup:
    return x_min[n*k:, :].reshape(x.shape)
  else:
    return x_min.reshape(x.shape)
