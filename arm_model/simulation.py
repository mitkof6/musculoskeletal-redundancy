import sympy as sp
import numpy as np
import pylab as plt
from scipy.integrate import odeint
from logger import Logger

# ------------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------------


class Simulation:
    """This class accepts a ArmModel and a Controller class that must define a
    function control(x, t, x0) and performs a numerical integration.

    """

    def __init__(self, model, controller):
        """ Constructor.

        Parameters
        ----------

        model: a reference to ToyModel

        controller: a reference to a tracking controller (forcing)

        """
        self.logger = Logger('Simulation')
        self.model = model
        # initial state
        self.x0 = model.state0
        # check if simulation has finished
        self.finished = False
        self.controller = controller

    def integrate(self, tf):
        """Numerical integration of the model for the given initial state the provided
        controller and target.

        Parameters
        ----------

        tf: simulation end time

        """
        # simulation parameters
        self.fps = 30
        self.tf = tf
        self.t = np.linspace(0.0, self.tf, self.tf * self.fps)

        def dxdt(x, t):
            return self.model.rhs(x, t,
                                  self.controller.controller(x, t, self.x0),
                                  self.model.constants)

        # integrate the system
        self.logger.debug('Integrating ...')
        self.finished = False
        self.x = odeint(dxdt, self.x0, self.t)
        self.finished = True
        self.logger.debug('Integration finished: ' + str(self.x.shape))

    def plot_simulation(self, ax=None):
        """Visualize initial and final pose along with the state variables.

        Parameters
        ----------

        ax: 1 x 3 axis

        """
        if not self.finished:
            self.logger('Simulation has not been performed')
            return

        if ax is None or ax.shape[0] < 3:
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        # state
        x = self.x
        t = self.t
        n = self.model.nd
        ax[0].plot(t, np.rad2deg(x[:, :n]))
        ax[0].legend(["${}$".format(sp.physics.vector.vlatex(temp))
                      for temp in self.model.coordinates])
        ax[0].set_xlabel('$t \; (s)$')
        ax[0].set_ylabel('$q \; (deg)$')
        ax[0].set_title('Generalized Coordinates')

        # generalized speeds u's
        ax[1].plot(t, np.rad2deg(x[:, n:]))
        ax[1].legend(["${}$".format(sp.physics.vector.vlatex(temp))
                      for temp in self.model.speeds])
        ax[1].set_xlabel('$t \; (s)$')
        ax[1].set_ylabel('$u \; (deg/s)$')
        ax[1].set_title('Generalized Speeds')

        # initial and final pose
        poseA = x[0, :n].tolist()
        poseB = x[-1, :n].tolist()
        self.model.draw_model(poseA, False, ax=ax[2], scale=0.7,
                              use_axis_limits=True, alpha=0.3, text=False)
        self.model.draw_model(poseB, False, ax=ax[2], scale=0.7,
                              use_axis_limits=True, alpha=1.0, text=False)

# ------------------------------------------------------------------------
# SimulationReporter
# ------------------------------------------------------------------------


class SimulationReporter:
    """A generic simulation reporter.

    """

    def __init__(self, model):
        """Constructor

        Parameters
        ----------
        model: a reference to ToyModel

        """
        self.logger = Logger('SimulationReporter')
        self.model = model
        self.t = []
        self.q = []
        self.u = []
        self.qd = []
        self.tau = []
        self.x = []
        self.xd = []
        self.lm = []
        self.lm_d = []
        self.lm_del = []
        self.lmd = []
        self.lmd_d = []
        self.fm = []
        self.ft = []

    def plot_joint_space_data(self, ax=None):
        """Plots joint space data. [Torques, Coordinates]

        Parameters
        ----------

        ax: axis must be of dim >= 2

        """

        n = self.model.nd
        t = np.asarray(self.t)
        q = np.asarray(self.q)
        qd = np.asarray(self.qd)
        tau = np.asarray(self.tau)

        if ax is None or ax.shape[0] < 2:
            fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True)

        # torques
        ax[0].plot(t, tau[:, :])
        ax[0].legend(["${}$".format(sp.physics.vector.vlatex(temp))
                      for temp in self.model.specifieds])
        ax[0].set_xlabel('$t \; (s)$')
        ax[0].set_ylabel('$\\tau \; (Nm)$')
        ax[0].set_title('Generalized Forces')

        # coordinates
        ax[1].plot(t, np.rad2deg(q[:, :]))
        ax[1].set_title('Joint Space Goals')
        ax[1].plot(t, np.rad2deg(qd[:, :]))
        ax[1].legend(['$q_' + str(i) + '$' for i in range(1, n + 1)] +
                     ['$q^d_' + str(i) + '$' for i in range(1, n + 1)])
        ax[1].set_xlabel('$t \; (s)$')
        ax[1].set_ylabel('$q \; (deg)$')

    def plot_task_space_data(self, ax=None):
        """Plots task space data. [Torques, Task Positions]

        Parameters
        ----------

        ax: axis must be of dim >= 3

        """
        t = np.asarray(self.t)
        x = np.asarray(self.x)
        xd = np.asarray(self.xd)
        ft = np.asarray(self.ft)
        tau = np.asarray(self.tau)

        if ax is None or ax.shape[0] < 3:
            fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True)

        # torques
        ax[0].plot(t, tau[:, :])
        ax[0].legend(["${}$".format(sp.physics.vector.vlatex(temp))
                      for temp in self.model.specifieds])
        ax[0].set_xlabel('$t \; (s)$')
        ax[0].set_ylabel('$\\tau \; (Nm)$')
        ax[0].set_title('Generalized Forces')

        # task positions
        ax[1].plot(t, x[:, :])
        ax[1].plot(t, xd[:, :])
        ax[1].legend(['$x_s$', '$y_s$', '$x_d$', '$y_d$'])
        ax[1].set_xlabel('$t \; (s)$')
        ax[1].set_ylabel('$x \; (m)$')
        ax[1].set_title('Task Space Goals')

        # task space forces
        ax[2].plot(t, ft[:, :])
        ax[2].legend(['$f_{t' + str(i) + '}$' for i in ['x', 'y']])
        ax[2].set_xlabel('$t \; (s)$')
        ax[2].set_ylabel('$f_t \; (N)$')
        ax[2].set_title('Task Space Forces')

    def plot_muscle_space_data_js(self, ax=None):
        """Plots muscle space controller reporter. [Torques]

        Parameters
        ----------

        ax: axis must be of dim >= 3

        """
        n = self.model.nd
        m = self.model.md
        t = np.asarray(self.t)
        q = np.asarray(self.q)
        qd = np.asarray(self.qd)
        lmd = np.asarray(self.lmd)
        lmd_d = np.asarray(self.lmd_d)
        tau = np.asarray(self.tau)

        if ax is None or ax.shape[0] < 3:
            fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True)

        # torques
        ax[0].plot(t, tau[:, :])
        ax[0].legend(["${}$".format(sp.physics.vector.vlatex(temp))
                      for temp in self.model.specifieds])
        ax[0].set_xlabel('$t \; (s)$')
        ax[0].set_ylabel('$\\tau \; (Nm)$')
        ax[0].set_title('Generalized Forces')

        # coordinates
        ax[1].plot(t, np.rad2deg(q[:, :]))
        ax[1].set_title('Goals')
        ax[1].plot(t, np.rad2deg(qd[:, :]))
        ax[1].legend(['$q_' + str(i) + '$' for i in range(1, n + 1)] +
                     ['$q^d_' + str(i) + '$' for i in range(1, n + 1)])
        ax[1].set_xlabel('$t \; (s)$')
        ax[1].set_ylabel('$q \; (deg)$')

        # muscle lengths
        ax[2].plot(t, lmd[:, :])
        ax[2].plot(t, lmd_d[:, :])
        ax[2].legend(['$\dot{l}_{m' + str(i) + '}$' for i in range(1, m + 1)] +
                     ['$\dot{l}^d_{m' + str(i) + '}$' for i in range(1, m + 1)])
        ax[2].set_xlabel('$t \; (s)$')
        ax[2].set_ylabel('$\dot{l}_m \; (m / s)$')
        ax[2].set_title('Muscle Lengthening (+) Shortening (-)')

    def plot_postural_muscle_space_data(self, ax=None):
        """Plots postural muscle space controller reporter. [Torques, Muslce Lengths]

        Parameters
        ----------

        ax: axis must be of dim >= 3

        """
        m = self.model.md
        t = np.asarray(self.t)
        lm = np.asarray(self.lm)
        lm_d = np.asarray(self.lm_d)
        fm = np.asarray(self.fm)
        tau = np.asarray(self.tau)

        if ax is None or ax.shape[0] < 3:
            fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True)

        # torques
        ax[0].plot(t, tau[:, :])
        ax[0].legend(["${}$".format(sp.physics.vector.vlatex(temp))
                      for temp in self.model.specifieds])
        ax[0].set_xlabel('$t \; (s)$')
        ax[0].set_ylabel('$\\tau \; (Nm)$')
        ax[0].set_title('Generalized Forces')

        # muscle lengths
        ax[1].plot(t, lm[:, :], '--')
        ax[1].set_prop_cycle(None)
        ax[1].plot(t, lm_d[:, :])
        # ax[1].legend(['$l_{m' + str(i) + '}$' for i in range(1, m + 1)] +
        #              ['$l^d_{m' + str(i) + '}$' for i in range(1, m + 1)])
        ax[1].set_xlabel('$t \; (s)$')
        ax[1].set_ylabel('$l_m \; (m)$')
        ax[1].set_title('Muscle Length Goals')

        # muscle forces
        ax[2].plot(t, fm[:, :])
        ax[2].legend(['$f_{m_{' + str(i) + '}}$' for i in range(1, m + 1)]
                     # + ['$l^d_{m' + str(i) + '}(t-\tau)$' for i in range(1, m + 1)]
                     )
        ax[2].set_xlabel('$t \; (s)$')
        ax[2].set_ylabel('$f_m \; (N)$')
        ax[2].set_title('Muscle Forces')
