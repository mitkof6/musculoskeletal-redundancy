import sympy as sp
import numpy as np
from logger import Logger
from util import to_np_mat
from analysis import construct_muscle_space_inequality
from scipy.optimize import minimize

# ------------------------------------------------------------------------
# Task Space
# ------------------------------------------------------------------------


class TaskSpace:
    """Task space EoM of a given task.

    Assuming the following joint space convention:

    M qddot + f = tau

    task projection is given by:

    ft = Lt (xddot - JDotQDot) + JtBarT f

    Lt = (R M^{-1} R^T)^{-1}

    JtBarT = Lt R M^{-1}

    NtT = (I - JtT JtBarT)

    and the resultant task space forces are given by:

    tau = JtT ft + NtT f

    """

    def __init__(self, model, xt):
        """ Constructor.

        Parameters
        ----------

        model: model
        xt: [d x 1] task positions

        """
        self.logger = Logger('TaskSpace')
        self.model = model
        self.xt = xt
        self.__construct_task_space_variables()

    def __construct_task_space_variables(self):
        """Calculate task Jacobian Jt and JtDot * qDot

        """
        self.Jt = self.xt.jacobian(self.model.Q())
        self.JtT = self.Jt.transpose()
        JtDot = sp.diff(self.Jt, self.model.t)
        self.JtDotQDot = JtDot * sp.Matrix(self.model.U())
        # the derivative of a matrix with a vector is a rank 3 tensor (3D
        # array), [dM/dq1, dM/dq2, ...]
        self.JtTDq = sp.derive_by_array(self.JtT, self.model.Q())

    def calculate_force(self, xddot, pose):
        """Calculate task forces.

        For a given end effector acceleration compute the joint space
        torques:

        ft = Lt (xddot - JDotQDot) + JtBarT f

        tau = JtT ft + NtT f

        Parameters
        ----------

        xddot: [2 x 1] a sympy Matrix containing the desired accelerations in 2D
        pose: system constants, coordinates and speeds as dictionary

        Returns
        -------

        (tau, ft): required torque and task forces to track the desired
        acceleration

        """
        M = to_np_mat(self.model.M.subs(pose))
        f = to_np_mat(self.model.f.subs(pose))
        Jt = to_np_mat(self.Jt.subs(pose))
        JtDotQDot = to_np_mat(self.JtDotQDot.subs(pose))
        JtT = to_np_mat(self.JtT.subs(pose))

        MInv = np.linalg.inv(M)
        LtInv = Jt * MInv * JtT
        Lt = np.linalg.pinv(LtInv)
        JtBarT = Lt * Jt * MInv
        NtT = np.asmatrix(np.eye(len(JtT))) - JtT * JtBarT

        ft = Lt * (xddot - JtDotQDot) + JtBarT * f

        return JtT * ft + 0 * NtT * f, ft

    def x(self, pose):
        """For a given pose (q) evaluate the task position.

        Parameters
        ----------

        pose: dictionary of model parameters and q's

        Returns
        -------

        x: sympy Matrix

        """
        return self.xt.subs(pose)

    def u(self, pose, qdot):
        """For a given pose (q, u) evaluate the task velocity.

        Parameters
        ----------

        pose: dictionary of model parameters, q's and u's
        qdot: an array of u(t)

        Returns
        ------

        u: sympy Matrix

        """
        return self.Jt.subs(pose) * sp.Matrix(qdot)


# ------------------------------------------------------------------------
# Muscle Space
# ------------------------------------------------------------------------


class MuscleSpace:
    """ Muscle space EoM.
    """

    def __init__(self, model, use_optimization=False):
        """ Constructor

        Parameters
        ----------

        model: model

        """
        self.logger = Logger('MuscleSpace')
        self.model = model
        self.use_optimization = use_optimization
        self.Fmax = to_np_mat(self.model.Fmax)
        self.x_max = np.max(self.Fmax)

    def calculate_force(self, lmdd, pose):
        """Calculate muscle forces.

        For a given muscle length acceleration compuyte the muscle space EoM of
        motion and required muscle force to track the goal acceleration.

        fm_par = -Lm (lmdd - RDotQDot) - RBarT f

        tau = -R^T fm_par

        Parameters
        ----------

        lmdd: [m x 1] a sympy Matrix containing the desired muscle length
        accelerations

        pose: dictionary

        Returns
        -------

        (tau, fm_par + fm_perp) required torque to track the desired acceleration

        """

        M = to_np_mat(self.model.M.subs(pose))
        f = to_np_mat(self.model.f.subs(pose))
        R = to_np_mat(self.model.R.subs(pose))
        RT = R.transpose()
        RDotQDot = to_np_mat(self.model.RDotQDot.subs(pose))

        MInv = np.linalg.inv(M)
        LmInv = R * MInv * RT
        Lm = np.linalg.pinv(LmInv)
        RBarT = np.linalg.pinv(RT)
        NR = np.asmatrix(np.eye(len(RBarT)) - RBarT * RT)

        fm_par = -Lm * (lmdd - RDotQDot) - RBarT * f

        # Ensure fm_par > 0 not required for simulation, but for muscle analysis
        # otherwise muscle forces will be negative. Since RT * NR = 0 the null
        # space term does not affect the resultant torques.
        m = fm_par.shape[0]
        fm_0 = np.zeros((m, 1))
        if self.use_optimization:
            Z, B = construct_muscle_space_inequality(NR, fm_par, self.Fmax)

            def objective(x):
                return np.sum(x**2)

            def inequality_constraint(x):
                return np.array(B - Z * (x.reshape(-1, 1))).reshape(-1,)

            x0 = np.zeros(m)
            bounds = tuple([(-self.x_max, self.x_max) for i in range(0, m)])
            constraints = ({'type': 'ineq', 'fun': inequality_constraint})
            sol = minimize(objective, x0,  method='SLSQP',
                           bounds=bounds,
                           constraints=constraints)
            fm_0 = sol.x.reshape(-1, 1)
            if sol.success == False:
                raise RuntimeError('Some muscles are too week for this action')

        fm_perp = NR * fm_0

        return -RT * fm_par, fm_par + fm_perp
