import numpy as np
import pylab as plt
import matplotlib as mpl
# from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from util import to_np_mat, plot_corr_ellipses,  \
    convex_bounded_vertex_enumeration, nullspace
from logger import Logger


# ------------------------------------------------------------------------
# FeasibleMuscleSetAnalysis
# ------------------------------------------------------------------------

def construct_muscle_space_inequality(NR, fm_par, Fmax):
    """Construct the feasible muscle space Z f_m0 <= B .

    Parameters
    ----------

    NR: moment arm null space matrix

    fm_par: particular muscle forces

    Fmax: maximum muscle force

    """
    Z0 = -NR
    Z1 = NR
    B0 = fm_par
    B1 = np.asmatrix(np.diag(Fmax)).reshape(Fmax.shape[0], 1) - fm_par
    Z = np.concatenate((Z0, Z1), axis=0)
    B = np.concatenate((B0, B1), axis=0)
    return Z, B


class FeasibleMuscleSetAnalysis:
    """Feasible muscle set analysis.

    The required command along with the state of the system are recorded. Then
    this information is used to compute the feasible muscle null space and
    visualize it.

    """

    def __init__(self, model, simulation_reporter):
        """
        """
        self.logger = Logger('FeasibleMuscleSetAnalsysis')
        self.model = model
        self.simulation_reporter = simulation_reporter

    def visualize_simple_muscle(self, t, ax=None):
        """Visualize the feasible force set at a particular time instance for a linear
        muscle.

        Parameters
        ----------

        t: time

        ax: 1 x 3 axis

        """
        m = self.model.md
        q, Z, B, NR, fm_par = self.calculate_simple_muscles(t)
        x_max = np.max(to_np_mat(self.model.Fmax))
        fm_set = self.generate_solutions(Z, B, NR, fm_par)
        dataframe = pd.DataFrame(fm_set, columns=['$m_' + str(i) + '$' for i in
                                                  range(1, m + 1)])

        # box plot
        if ax is None or ax.shape[0] < 3:
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        # box plot
        dataframe.plot.box(ax=ax[0])
        ax[0].set_xlabel('muscle id')
        ax[0].set_ylabel('force $(N)$')
        ax[0].set_title('Muscle-Force Box Plot')
        ax[0].set_ylim([0, 1.1 * x_max])

        # correlation matrix
        cmap = mpl.cm.jet
        norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        corr = dataframe.corr()
        m = plot_corr_ellipses(corr, ax=ax[1], norm=norm, cmap=cmap)
        cb = plt.colorbar(m, ax=ax[1], orientation='vertical', norm=norm,
                          cmap=cmap)
        cb.set_label('Correlation Coefficient')
        ax[1].margins(0.1)
        ax[1].set_xlabel('muscle id')
        ax[1].set_ylabel('muscle id')
        ax[1].set_title('Correlation Matrix')
        ax[1].axis('equal')

        # draw model
        self.model.draw_model(q, False, ax[2], scale=0.7, text=False)

    def calculate_simple_muscles(self, t):
        """Construct Z f_m0 <= B for the case of a linear muscle model for a particular
        time instance.

        Parameters
        ----------

        t: time

        """
        # find nearesrt index corresponding to t
        idx = np.abs(np.array(self.simulation_reporter.t) - t).argmin()
        t = self.simulation_reporter.t[idx]
        q = self.simulation_reporter.q[idx]
        u = self.simulation_reporter.u[idx]
        tau = self.simulation_reporter.tau[idx]
        pose = self.model.model_parameters(q=q, u=u)
        n = self.model.nd

        # calculate required variables
        R = to_np_mat(self.model.R.subs(pose))
        RBarT = np.asmatrix(np.linalg.pinv(R.T))
        # reduce to independent columns to avoid singularities (proposition 3)
        NR = nullspace(R.transpose())
        fm_par = np.asmatrix(-RBarT * tau.reshape((n, 1)))
        Fmax = to_np_mat(self.model.Fmax)

        Z, B = construct_muscle_space_inequality(NR, fm_par, Fmax)

        return q, Z, B, NR, fm_par

    def generate_solutions(self, A, b, NR, fm_par):
        """Sample the solution space that satisfy A x <= b.

        Parameters
        ----------

        A: matrix A

        b: column vector

        NR: moment arm nullspace

        fm_par: particular solution

        Returns
        -------

        muscle forces: a set of solutions that satisfy the problem

        """
        feasible_set = []
        fm0_set = convex_bounded_vertex_enumeration(np.array(A),
                                                    np.array(b).flatten(), 0)
        n = fm0_set.shape[0]
        for i in range(0, n):
            fm = fm_par + NR * np.matrix(fm0_set[i, :]).reshape(-1, 1)
            feasible_set.append(fm)

        return np.array(feasible_set).reshape(n, -1)


def test_feasible_set(model):
    feasible_set = FeasibleMuscleSetAnalysis(model)
    n = model.nd
    m = model.md
    feasible_set.record(1,
                        np.random.random((m, 1)),
                        np.random.random((m, m)),
                        np.random.random((n, 1)))
    fig, ax = plt.subplots(2, 3, figsize=(10, 10))
    feasible_set.visualize_simple_muscle(1, ax[0])
    feasible_set.visualize_simple_muscle(1, ax[1])
    plt.show()
