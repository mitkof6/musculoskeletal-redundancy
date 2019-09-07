import os
import time
import sympy as sp
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import EllipseCollection
from matplotlib import patches
from fractions import Fraction

# ------------------------------------------------------------------------------
# logger
# ------------------------------------------------------------------------------

# logging.basicConfig(format='%(levelname)s %(asctime)s @%(name)s # %(message)s',
#                     datefmt='%m/%d/%Y %I:%M:%S %p',
#                     # filename='example.log',
#                     level=logging.DEBUG)

# ------------------------------------------------------------------------------
# utilities
# ------------------------------------------------------------------------------


def mat_show(mat):
    """Graphical visualization of a 2D matrix.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(to_np_mat(mat), interpolation='nearest')
    fig.colorbar(cax)


def is_symmetric(a, tol=1e-8):
    """Check if matrix is symmetric.

    """
    return np.allclose(a, a.T, atol=tol)


def mat(array):
    """For a given 2D array return a numpy matrix.
    """
    return np.matrix(array)


def vec(vector):
    """Construct a column vector of type numpy matrix.
    """
    return np.matrix(vector).reshape(-1, 1)


def to_np_mat(sympy_mat):
    """Cast sympy Matrix to numpy matrix of float type.

    Parameters
    ----------
    m: sympy 2D matrix

    Returns
    -------
    a numpy asmatrix

    """
    return np.asmatrix(sympy_mat.tolist(), dtype=np.float)


def to_np_array(sympy_mat):
    """Cast sympy Matrix to numpy matrix of float type. Works for N-D matrices as
    compared to to_np_mat().

    Parameters
    ----------
    m: sympy 2D matrix

    Returns
    -------
    a numpy asmatrix

    """
    return np.asarray(sympy_mat.tolist(), dtype=np.float)


def to_np_vec(sympy_vec):
    """Transforms a 1D sympy vector (e.g. 5 x 1) to numpy array (e.g. (5,)).

    Parameters
    ----------
    v: 1D sympy vector

    Returns
    -------
    a 1D numpy array

    """
    return np.asarray(sp.flatten(sympy_vec), dtype=np.float)


def is_pd(A):
    """Check if matrix is positive definite
    """
    return np.all(np.linalg.eigvals(A) > 0)


def nullspace(A, atol=1e-13, rtol=0):
    """Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    """

    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns


def lrs_inequality_vertex_enumeration(A, b):
    """Find the vertices given an inequality system A * x <= b. This function
    depends on lrs library.

    Parameters
    ----------

    A: numpy array [m x n]

    b: numpy array [m]

    Returns
    -------

    v: numpy array [k x n]
        the vertices of the polytope

    """
    # export H-representation
    with open('temp.ine', 'w') as file_handle:
        file_handle.write('Feasible_Set\n')
        file_handle.write('H-representation\n')
        file_handle.write('begin\n')
        file_handle.write(str(A.shape[0]) + ' ' +
                          str(A.shape[1] + 1) + ' rational\n')
        for i in range(0, A.shape[0]):
            file_handle.write(str(Fraction(b[i])))
            for j in range(0, A.shape[1]):
                file_handle.write(' ' + str(Fraction(-A[i, j])))

            file_handle.write('\n')

        file_handle.write('end\n')

    # call lrs
    try:
        os.system('lrs temp.ine > temp.ext')
    except OSError as e:
        raise RuntimeError(e)

    # read the V-representation
    vertices = []
    with open('temp.ext', 'r') as file_handle:
        begin = False
        for line in file_handle:
            if begin:
                if 'end' in line:
                    break

                comp = line.split()
                try:
                    v_type = comp.pop(0)
                except:
                    print('No feasible solution')

                if v_type is '1':
                    v = [float(Fraction(i)) for i in comp]
                    vertices.append(v)

            else:
                if 'begin' in line:
                    begin = True

    # delete temporary files
    try:
        os.system('rm temp.ine temp.ext')
    except OSError as e:
        pass

    return vertices


def ccd_inequality_vertex_enumeration(A, b):
    """Find the vertices given an inequality system A * x <= b. This function
    depends on pycddlib (cdd).

    Parameters
    ----------

    A: numpy array [m x n]

    b: numpy array [m]

    Returns
    -------

    v: numpy array [k x n]
        the vertices of the polytope

    """
    import cdd
    # try floating point, if problem fails try exact arithmetics (slow)
    try:
        M = cdd.Matrix(np.hstack((b.reshape(-1, 1), -A)),
                       number_type='float')
        M.rep_type = cdd.RepType.INEQUALITY
        p = cdd.Polyhedron(M)
    except:
        print('Warning: switch to exact arithmetics')
        M = cdd.Matrix(np.hstack((b.reshape(-1, 1), -A)),
                       number_type='fraction')
        M.rep_type = cdd.RepType.INEQUALITY
        p = cdd.Polyhedron(M)

    G = np.array(p.get_generators())

    if not G.shape[0] == 0:
        return G[np.where(G[:, 0] == 1.0)[0], 1:].tolist()
    else:
        raise ValueError('Infeasible Inequality')


def optimization_based_sampling(A, b, optimziation_samples,
                                closiness_tolerance, max_opt_iterations):
    """Efficient method for sampling the feasible set for a large system of
    inequalities (A x <= b). When the dimension of the set (x) is large,
    deterministic and pure randomized techniques fail to solve this problem
    efficiently.

    This method uses constrained optimization in order to find n solutions that
    satisfy the inequality. Each iteration new randomized objective function is
    assigned so that the optimization will find a different solution.

    Parameters
    ----------
    A: numpy array [m x n]

    b: numpy array [m]

    optimziation_samples: integer
        number of samples to generate

    closiness_tolerance: float
        accept solution if distance is larger than the provided tolerance

    max_opt_iterations: integer
        maximum iteration of the optimization algorithm

    Returns
    -------

    solutions: list

    """
    from scipy.optimize import minimize

    nullity = A.shape[1]
    solutions = []
    j = 0
    while j < optimziation_samples:
        # change objective function randomly (Dirichlet distribution ensures
        # that w sums to 1)
        w = np.random.dirichlet(np.ones(nullity), size=1)

        def objective(x):
            return np.sum((w * x) ** 2)

        # solve the optimization_based_sampling
        def inequality_constraint(x):
            return A.dot(x) - b

        x0 = np.random.uniform(-1, 1, nullity)
        bounds = tuple([(None, None) for i in range(0, nullity)])
        constraints = ({'type': 'ineq', 'fun': inequality_constraint})
        options = {'maxiter': max_opt_iterations}
        sol = minimize(objective, x0, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options=options)

        x = sol.x
        # check if solution satisfies the system
        if np.all(A.dot(x) <= b):
            # if solution is not close to the rest then accept
            close = False
            for xs in solutions:
                if np.linalg.norm(xs - x, 2) < closiness_tolerance:
                    close = True
                    break

            if not close:
                solutions.append(x)
                j = j + 1

    return solutions


def convex_bounded_vertex_enumeration(A,
                                      b,
                                      convex_combination_passes=1,
                                      method='lrs'):
    """Sample a convex, bounded inequality system A * x <= b. The vertices of the
    convex polytope are first determined. Then the convexity property is used to
    generate additional solutions within the polytope.

    Parameters
    ----------

    A: numpy array [m x n]

    b: numpy array [m]

    convex_combination_passes: int (default 1)
        recombine vertices to generate additional solutions using the convex
        property

    method: str (lrs or cdd or rnd)

    Returns
    -------

    v: numpy array [k x n]
        solutions within the convex polytope

    """
    # find polytope vertices
    if method == 'lrs':
        solutions = lrs_inequality_vertex_enumeration(A, b)
    elif method == 'cdd':
        solutions = ccd_inequality_vertex_enumeration(A, b)
    elif method == 'rnd':
        solutions = optimization_based_sampling(A, b,
                                                A.shape[0] ** 2,
                                                1e-3, 1000)
    else:
        raise RuntimeError('Unsupported method: choose "lrs" or "cdd" or "rnd"')

    # since the feasible space is a convex set we can find additional solution
    # in the form z = a * x_i + (1-a) x_j
    for g in range(0, convex_combination_passes):
        n = len(solutions)
        for i in range(0, n):
            for j in range(0, n):
                if i == j:
                    continue

                a = 0.5
                x1 = np.array(solutions[i])
                x2 = np.array(solutions[j])
                z = a * x1 + (1 - a) * x2
                solutions.append(z.tolist())

    # remove duplicates from 2D list
    solutions = [list(t) for t in set(tuple(element) for element in solutions)]

    return np.array(solutions, np.float)


def test_inequality_sampling(d):
    A = np.array([[0, 0, -1],
                  [0, -1, 0],
                  [1, 0, 0],
                  [-1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    b = 0.5 * np.ones(6)
    print(A, b)
    t1 = time.time()
    # solutions = optimization_based_sampling(A, b, 20, 0.3, 1000)
    solutions = convex_bounded_vertex_enumeration(A, b, d)
    t2 = time.time()
    print('Execution time: ' + str(t2 - t1) + 's')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(solutions[:, 0], solutions[:, 1], solutions[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    fig.tight_layout()
    fig.savefig('results/inequality_sampling/inequality_sampling_d' + str(d) + '.png',
                format='png', dpi=300)
    fig.savefig('results/inequality_sampling/inequality_sampling_d' + str(d) + '.pdf',
                format='pdf', dpi=300)
    fig.savefig('results/inequality_sampling/inequality_sampling_d' + str(d) + '.eps',
                format='eps', dpi=300)


# test_inequality_sampling(0)
# test_inequality_sampling(1)
# test_inequality_sampling(2)

def tensor3_vector_product(T, v):
    """Implements a product of a rank-3 tensor (3D array) with a vector using
    tensor product and tensor contraction.

    Parameters
    ----------

    T: sp.Array of dimensions n x m x k

    v: sp.Array of dimensions k x 1

    Returns
    -------

    A: sp.Array of dimensions n x m

    Example
    -------

    >>>T = sp.Array([[[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]],
                     [[13, 16, 19, 22], [14, 17, 20, 23], [15, 18, 21, 24]]])
    ⎡⎡1  4  7  10⎤  ⎡13  16  19  22⎤⎤
    ⎢⎢           ⎥  ⎢              ⎥⎥
    ⎢⎢2  5  8  11⎥  ⎢14  17  20  23⎥⎥
    ⎢⎢           ⎥  ⎢              ⎥⎥
    ⎣⎣3  6  9  12⎦  ⎣15  18  21  24⎦⎦
    >>>v = sp.Array([1, 2, 3, 4]).reshape(4, 1)
    ⎡1⎤
    ⎢ ⎥
    ⎢2⎥
    ⎢ ⎥
    ⎢3⎥
    ⎢ ⎥
    ⎣4⎦
    >>>tensor3_vector_product(T, v)
    ⎡⎡70⎤  ⎡190⎤⎤
    ⎢⎢  ⎥  ⎢   ⎥⎥
    ⎢⎢80⎥  ⎢200⎥⎥
    ⎢⎢  ⎥  ⎢   ⎥⎥
    ⎣⎣90⎦  ⎣210⎦⎦

    """
    import sympy as sp
    assert(T.rank() == 3)
    # reshape v to ensure 1D vector so that contraction do not contain x 1
    # dimension
    v.reshape(v.shape[0], )
    p = sp.tensorproduct(T, v)
    return sp.tensorcontraction(p, (2, 3))


def test_tensor_product():
    T = sp.Array([[[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]],
                  [[13, 16, 19, 22], [14, 17, 20, 23], [15, 18, 21, 24]]])
    v = sp.Array([1, 2, 3, 4]).reshape(4, 1)
    display(T, v)
    display(tensor3_vector_product(T, v))


# test_tensor_product()


def draw_ellipse(ax, xc, A, scale=1.0, show_axis=False):
    """Construct an ellipse representation of a 2x2 matrix.

    Parameters
    ----------
    ax: plot axis

    xc: np.array 2 x 1
        center of the ellipse
    mat: np.array 2 x 2

    scale: float (default=1.0)
        scale factor of the principle axes

    """
    eigen_values, eigen_vectors = np.linalg.eig(A)
    idx = np.abs(eigen_values).argsort()[::-1]
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]
    phi = np.rad2deg(np.arctan2(eigen_vectors[1, 0], eigen_vectors[0, 0]))

    ellipse = patches.Ellipse(xy=(xc[0, 0], xc[1, 0]),
                              width=2 * scale * eigen_values[0],
                              height=2 * scale * eigen_values[1],
                              angle=phi,
                              linewidth=2, fill=False)
    ax.add_patch(ellipse)

    # axis
    if show_axis:
        x_axis = np.array([[xc[0, 0], xc[1, 0]],
                           [xc[0, 0] + scale * np.abs(eigen_values[0]) * eigen_vectors[0, 0],
                            xc[1, 0] + scale * np.abs(eigen_values[0]) * eigen_vectors[1, 0]]])
        y_axis = np.array([[xc[0, 0], xc[1, 0]],
                           [xc[0, 0] + scale * eigen_values[1] * eigen_vectors[0, 1],
                            xc[1, 0] + scale * eigen_values[1] * eigen_vectors[1, 1]]])
        ax.plot(x_axis[:, 0], x_axis[:, 1], '-r', label='x-axis')
        ax.plot(y_axis[:, 0], y_axis[:, 1], '-g', label='y-axis')

    return phi, eigen_values, eigen_vectors


def test_ellipse():
    fig, ax = plt.subplots()
    xc = vec([0, 0])
    M = mat([[2, 1], [1, 2]])
    # M = mat([[-2.75032375, -11.82938331], [-11.82938331, -53.5627191]])
    print(np.linalg.matrix_rank(M))
    phi, l, v = draw_ellipse(ax, xc, M, 1, True)
    print(phi, l, v)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.legend()
    fig.show()


# test_ellipse()


def calculate_feasible_muscle_set(feasible_muscle_set_analysis, base_name,
                                  t_start, t_end, dt, speed):
    """ Calculates the feasible muscle space of a simulation.

    Parameters
    ----------

    feasible_muscle_set_analysis: FeasibleMuscleSetAnalysis

    base_name: base name of simulation files

    t_start: t start

    t_end: t end

    dt: time interval for reporting

    speed: speed of animation

    """
    print('Calculating feasible muscle set ...')
    time = np.linspace(t_start, t_end, t_end / dt + 1, endpoint=True)
    for i, t in enumerate(tqdm(time)):
        visualize_feasible_muscle_set(feasible_muscle_set_analysis, t,
                                      base_name + str(i).zfill(6), 'png')

    command = 'convert -delay ' + \
              str(speed * dt) + ' -loop 0 ' + base_name + \
              '*.png ' + base_name + 'anim.gif'

    print(command)
    try:
        os.system(command)
    except:
        print('unable to execute command')


def visualize_feasible_muscle_set(feasible_muscle_set_analysis, t,
                                  fig_name='fig/feasible_muscle_set', format='png'):
    """ Visualization of the feasible muscle space.

    Parameters
    ----------

    feasible_muscle_set_analysis: FeasibleMuscleSetAnalysis
    t: time instance to evaluate the feasible
    fig_name: figure name for saving
    format: format (e.g. .png, .pdf, .eps)

    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    feasible_muscle_set_analysis.visualize_simple_muscle(t, ax)
    fig.suptitle('t = ' + str(np.around(t, decimals=2)),
                 y=1.00, fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(fig_name + '.' + format, format=format, dpi=300)
    fig.savefig(fig_name + '.pdf', format='pdf', dpi=300)
    fig.savefig(fig_name + '.eps', format='eps', dpi=300)


def apply_generalized_force(f):
    """Applies a generalized force (f) in a manner that is consistent with Newton's
    3rd law.

    Parameters
    ----------
    f: generalized force

    """
    n = len(f)
    tau = []
    for i in range(0, n):
        if i == n - 1:
            tau.append(f[i])
        else:
            tau.append(f[i] - f[i + 1])

    return tau


def custom_exponent(q, A, k, q_lim):
    """ Sympy representation of custom exponent function.

    f(q) = A e^(k (q - q_lim)) / (150) ** k

    """
    return A * sp.exp(k * (q - q_lim)) / (148.42) ** k


def coordinate_limiting_force(q, q_low, q_up, a, b):
    """A continuous coordinate limiting force for a rotational joint.

    It applies an exponential force when approximating a limit. The convention
    is that positive force is generated when approaching the lower limit and
    negative when approaching the upper. For a = 1, F ~= 1N at the limits.

    Parameters
    ----------
    q: generalized coordinate
    q_low: lower limit
    q_up: upper limit
    a: force at limits
    b: rate of the slop

    Note: q, q_low, q_up must have the same units (e.g. rad)

    """
    return custom_exponent(q_low + 5, a, b, q) - custom_exponent(q, a, b, q_up - 5)


def test_limiting_force():
    """
    """
    q = np.linspace(0, np.pi / 4, 100, endpoint=True)
    f = [coordinate_limiting_force(qq, 0, np.pi / 4, 1, 50) for qq in q]

    plt.plot(q, np.array(f))
    plt.show()


def gaussian(x, a, m, s):
    """Gaussian function.

    f(x) = a e^(-(x - m)^2 / (2 s ^2))

    Parameters
    ----------
    x: x
    a: peak
    m: mean
    s: standard deviation

    For a good approximation of an impulse at t = 0.3 [x, 1, 0.3, 0.01].

    """
    return a * np.exp(- (x - m) ** 2 / (2 * s ** 2))


def test_gaussian():
    """
    """
    t = np.linspace(0, 2, 200)
    y = [gaussian(tt, 0.4, 0.4, 0.01) for tt in t]
    plt.plot(t, y)
    plt.show()


def rotate(origin, point, angle):
    """Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.

    """
    R = np.asmatrix([[np.cos(angle), - np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])

    q = origin + R * (point - origin)

    return q


def sigmoid(t, t0, A, B):
    """Implementation of smooth sigmoid function.

    Parameters
    ----------
    t: time to be evalutaed
    t0: delay
    A: magnitude
    B: slope

    Returns
    -------
    (y, y', y'')

    """
    return (A * (np.tanh(B * (t - t0)) + 1) / 2,
            A * B * (- np.tanh(B * (t - t0)) ** 2 + 1) / 2,
            - A * B ** 2 * (- np.tanh(B * (t - t0)) ** 2 + 1) * np.tanh(B * (t - t0)))


def test_sigmoid():
    """
    """
    t, A, B, t0 = sp.symbols('t A B t0')
    y = A / 2 * (sp.tanh(B * (t - t0 - 1)) + 1)
    yd = sp.diff(y, t)
    ydd = sp.diff(yd, t)
    print('\n', y, '\n', yd, '\n', ydd)

    tt = np.linspace(-2, 2, 100)
    yy = np.array([sigmoid(x, 0.5, 2, 5) for x in tt])
    plt.plot(tt, yy)
    plt.show()


def plot_corr_ellipses(data, ax=None, **kwargs):
    """For a given correlation matrix "data", plot the correlation matrix in terms
    of ellipses.

    parameters
    ----------
    data: Pandas dataframe containing the correlation of the data (df.corr())
    ax: axis (e.g. fig, ax = plt.subplots(1, 1))
    kwards: keywords arguments (cmap="Greens")

    https://stackoverflow.com/questions/34556180/
    how-can-i-plot-a-correlation-matrix-as-a-set-of-ellipses-similar-to-the-r-open

    """
    M = np.array(data)
    if not M.ndim == 2:
        raise ValueError('data must be a 2D array')
    if ax is None:
        fig, ax = plt.subplots(1, 1, subplot_kw={'aspect': 'equal'})
        ax.set_xlim(-0.5, M.shape[1] - 0.5)
        ax.set_ylim(-0.5, M.shape[0] - 0.5)

    # xy locations of each ellipse center
    xy = np.indices(M.shape)[::-1].reshape(2, -1).T

    # set the relative sizes of the major/minor axes according to the strength
    # of the positive/negative correlation
    w = np.ones_like(M).ravel()
    h = 1 - np.abs(M).ravel()
    a = 45 * np.sign(M).ravel()

    ec = EllipseCollection(widths=w, heights=h, angles=a, units='x', offsets=xy,
                           transOffset=ax.transData, array=M.ravel(), **kwargs)
    ax.add_collection(ec)

    # if data is a DataFrame, use the row/column names as tick labels
    if isinstance(data, pd.DataFrame):
        ax.set_xticks(np.arange(M.shape[1]))
        ax.set_xticklabels(data.columns, rotation=90)
        ax.set_yticks(np.arange(M.shape[0]))
        ax.set_yticklabels(data.index)

    return ec


def get_cmap(n, name='hsv'):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct RGB
    color; the keyword argument name must be a standard mpl colormap name.

    """
    return plt.cm.get_cmap(name, n)


def assert_if_same(A, B):
    """Assert whether two quantities (value, vector, matrix) are the same."""
    assert np.isclose(
        np.array(A).astype(np.float64),
        np.array(B).astype(np.float64)).all() == True, 'quantities not equal'


def christoffel_symbols(M, q, i, j, k):
    """
    M [n x n]: inertia mass matrix
    q [n x 1]: generalized coordinates
    i, j, k  : the indexies to be computed
    """
    return sp.Rational(1, 2) * (sp.diff(M[i, j], q[k]) + sp.diff(M[i, k], q[j]) -
                                sp.diff(M[k, j], q[i]))


def coriolis_matrix(M, q, dq):
    """
    Coriolis matrix C(q, qdot) [n x n]
    Coriolis forces are computed as  C(q, qdot) *  qdot [n x 1]
    """
    n = M.shape[0]
    C = sp.zeros(n, n)
    for i in range(0, n):
        for j in range(0, n):
            for k in range(0, n):
                C[i, j] = C[i, j] + christoffel_symbols(M, q, i, j, k) * dq[k]
    return C


def substitute(symbols, constants):
    """For a given symbolic sequence substitute symbols."""
    return np.array([substitute(exp, constants)
                     if hasattr(exp, '__iter__')
                     else exp.subs(constants) for exp in symbols])
