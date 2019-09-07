# A variety of useful utilities.
#
# @author Dimitar Stanev (stanev@ece.upatras.gr)
import os
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# from mpl_toolkits.mplot3d import Axes3D
from fractions import Fraction


def plot_sto(sto_file, plots_per_row, plot_file, pattern=None,
             title_function=lambda x: x):
    """Plots the .sto file (OpenSim) by constructing a grid of subplots.

    Parameters
    ----------
    sto_file: str
        path to file
    plots_per_row: int
        subplot columns
    plot_file: str
        path to store results
    pattern: str, optional, default=None
        plot based on pattern (e.g. only pelvis coordinates)
    title_function: lambda
        callable function f(str) -> str
    """
    assert('pdf' in plot_file)

    header, labels, data = readMotionFile(sto_file)
    data = np.array(data)
    indices = []
    if pattern is not None:
        indices = index_containing_substring(labels, pattern)
    else:
        indices = range(1, len(labels))

    n = len(indices)
    ncols = int(plots_per_row)
    nrows = int(np.ceil(float(n) / plots_per_row))
    pages = int(np.ceil(float(nrows) / ncols))
    if ncols > n:
        ncols = n

    with PdfPages(plot_file) as pdf:
        for page in range(0, pages):
            fig, ax = plt.subplots(nrows=ncols, ncols=ncols,
                                   figsize=(8, 8))
            ax = ax.flatten()
            for pl, col in enumerate(indices[page * ncols ** 2:page *
                                             ncols ** 2 + ncols ** 2]):
                ax[pl].plot(data[:, 0], data[:, col])
                ax[pl].set_title(title_function(labels[col]))

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close()


def readMotionFile(filename):
    """Reads OpenSim .sto files.

    Parameters
    ----------
    filename: str
        absolute path to the .sto file

    Returns
    -------
    header: list of str
        the header of the .sto
    labels: list of str
        the labels of the columns
    data: list of lists
        an array of the data

    """

    if not os.path.exists(filename):
        print('file do not exists')

    file_id = open(filename, 'r')

    # read header
    next_line = file_id.readline()
    header = [next_line]
    nc = 0
    nr = 0
    while not 'endheader' in next_line:
        if 'datacolumns' in next_line:
            nc = int(next_line[next_line.index(' ') + 1:len(next_line)])
        elif 'datarows' in next_line:
            nr = int(next_line[next_line.index(' ') + 1:len(next_line)])
        elif 'nColumns' in next_line:
            nc = int(next_line[next_line.index('=') + 1:len(next_line)])
        elif 'nRows' in next_line:
            nr = int(next_line[next_line.index('=') + 1:len(next_line)])

        next_line = file_id.readline()
        header.append(next_line)

    # process column labels
    next_line = file_id.readline()
    if next_line.isspace() == True:
        next_line = file_id.readline()

    labels = next_line.split()

    # get data
    data = []
    for i in range(1, nr + 1):
        d = [float(x) for x in file_id.readline().split()]
        data.append(d)

    file_id.close()

    return header, labels, data


def index_containing_substring(list_str, pattern):
    """For a given list of strings finds the index of the element that contains the
    substring.

    Parameters
    ----------
    list_str: list of str

    pattern: str
         pattern


    Returns
    -------
    indices: list of int
         the indices where the pattern matches

    """
    indices = []
    for i, s in enumerate(list_str):
        if pattern in s:
            indices.append(i)

    return indices


def write_as_sto_file(file_name, labels, time, data):
    """Writes data to OpenSim .sto file format.

    Parameters
    ----------

    file_name: string

    labels: list string

    time: numpy array

    data: numpy array
    """
    n = time.shape[0]
    m = data.shape[1]
    assert (len(labels) == m + 1)

    with open(file_name, 'w') as handle:
        handle.write(
            'ModelForces\nversion=1\nnRows=%d\nnColumns=%d\ninDegrees=no\nendheader\n'
            % (n, m + 1))

        handle.write(labels[0])
        for i in range(1, len(labels)):
            handle.write('\t' + str(labels[i]))

        handle.write('\n')

        for i in range(0, n):
            handle.write('\t' + str(time[i]))
            for j in range(0, m):
                handle.write('\t' + str(data[i, j]))

            handle.write('\n')


def mat_show(mat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(to_np_mat(mat), interpolation='nearest')
    fig.colorbar(cax)


def mat(array):
    """For a given 2D array return a numpy matrix.
    """
    return np.matrix(array)


def vec(vector):
    """Construct a column vector of type numpy matrix.
    """
    return np.matrix(vector).reshape(-1, 1)


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
        file_handle.write(
            str(A.shape[0]) + ' ' + str(A.shape[1] + 1) + ' rational\n')
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
                v_type = comp.pop(0)
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
        M = cdd.Matrix(np.hstack((b.reshape(-1, 1), -A)), number_type='float')
        M.rep_type = cdd.RepType.INEQUALITY
        p = cdd.Polyhedron(M)
    except:
        print('Warning: switch to exact arithmetics')
        M = cdd.Matrix(
            np.hstack((b.reshape(-1, 1), -A)), number_type='fraction')
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
    max_infeasible = 100
    infeasible_counter = max_infeasible
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
            infeasible_counter = max_infeasible
            # if solution is not close to the rest then accept
            close = False
            for xs in solutions:
                if np.linalg.norm(xs - x, 2) < closiness_tolerance:
                    close = True
                    break

            if not close:
                solutions.append(x)
                j = j + 1

        else:
            infeasible_counter = infeasible_counter - 1
            if infeasible_counter == 0:
                raise RuntimeError('inequality is infeasible')

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
        raise NotImplementedError('Not fully tested yet')
        solutions = optimization_based_sampling(A, b,
                                                (A.shape[0]) ** 2,
                                                1e-1, 3000)
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
    import time
    from mpl_toolkits.mplot3d import Axes3D
    A = np.array([[0, 0, -1],
                  [0, -1, 0],
                  [1, 0, 0],
                  [-1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    b = 0.5 * np.ones(6)
    print(A, b)
    t1 = time.time()
    solutions = convex_bounded_vertex_enumeration(A, b, d, method='rnd')
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
    # fig.savefig('results/inequality_sampling/inequality_sampling_d' + str(d) + '.png',
    #             format='png', dpi=300)
    # fig.savefig('results/inequality_sampling/inequality_sampling_d' + str(d) + '.pdf',
    #             format='pdf', dpi=300)
    # fig.savefig('results/inequality_sampling/inequality_sampling_d' + str(d) + '.eps',
    #             format='eps', dpi=300)


# test_inequality_sampling(0)
# test_inequality_sampling(1)
# test_inequality_sampling(2)


def construct_muscle_space_inequality(NR, fm_par, fmax):
    """Construct the feasible muscle space Z f_m0 <= B.

    Parameters
    ----------

    NR: moment arm null space matrix

    fm_par: particular muscle forces

    fmax: maximum muscle force

    """
    Z0 = -NR
    Z1 = NR
    b0 = fm_par.reshape(-1, 1)
    b1 = fmax.reshape(-1, 1) - fm_par.reshape(-1, 1)
    Z = np.concatenate((Z0, Z1), axis=0)
    b = np.concatenate((b0, b1), axis=0)
    return Z, b


def null_space(A, atol=1e-13, rtol=0):
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


def simbody_matrix_to_list(M):
    """ Convert simbody Matrix to python list.

    Parameters
    ----------

    M: opensim.Matrix

    """
    return [[M.get(i, j) for j in range(0, M.ncol())]
            for i in range(0, M.nrow())]


def cartesian(arrays, out=None):
    """Generate a cartesian product of input arrays.

    Parameters
    ----------

    arrays: list of array-like
        1-D arrays to form the cartesian product of.

    out: ndarray
        Array to place the cartesian product in.

    Returns
    -------

    out: ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------

    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in xrange(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]

    return out


def construct_coordinate_grid(model, coordinates, N=5):
    """Given n coordinates get the coordinate range and generate a coordinate grid
    of combinations using cartesian product.

    Parameters
    ----------

    model: opensim.Model

    coordinates: list of string

    N: int (default=5)
        the number of points per coordinate

    Returns
    -------

    sampling_grid: np.array
        all combination of coordinates

    """
    sampling_grid = []
    for coordinate in coordinates:
        min_range = model.getCoordinateSet().get(coordinate).getRangeMin()
        max_range = model.getCoordinateSet().get(coordinate).getRangeMax()
        sampling_grid.append(
            np.linspace(min_range, max_range, N, endpoint=True))

    return cartesian(sampling_grid)


def find_intermediate_joints(origin_body, insertion_body, model_tree, joints):
    """Finds the intermediate joints between two bodies.

    Parameters
    ----------

    origin_body: string
        first body in the model tree

    insertion_body: string
        last body in the branch

    model_tree: list of dictionary relations {parent, joint, child}

    joints: list of strings
        intermediate joints
    """
    if origin_body == insertion_body:
        return True

    children = filter(lambda x: x['parent'] == origin_body, model_tree)
    for child in children:
        found = find_intermediate_joints(child['child'], insertion_body,
                                         model_tree, joints)
        if found:
            joints.append(child['joint'])
            return True

    return False
