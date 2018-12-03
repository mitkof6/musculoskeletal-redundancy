import sympy as sp
# import scipy as scp
import numpy as np
import pylab as plt
from sympy import cos, sin
from sympy.physics.mechanics import dynamicsymbols
from pydy.codegen.ode_function_generators import generate_ode_function
from util import coordinate_limiting_force, coriolis_matrix, \
    apply_generalized_force, substitute
from logger import Logger
from functools import reduce

# ------------------------------------------------------------------------
# ArmModel
# ------------------------------------------------------------------------


class ArmModel:
    """A planar toy model composed of 3 degrees of freedom and 9 muscles.

    The forces that act on the model are: gravity (tau_g), Coriolis and
    centrifugal (tau_c), coordinate limiting forces (tau_l) and viscous joints
    (tau_b). We assume the following:

    M qddot + tau_c + tau_g = tau + tau_l + tau_b

    M qddot = forcing

    forcing = tau - f, f = tau_c + tau_g - tau_l - tau_b

    or

    M qddot + f = tau

    Furthermore, we use 9 linear muscles. The musculotendon lengths and their
    derivatives are given by lm, lmd, lmdd. The muscle's moment arm matrix (R)
    is given by:

    R =  d lm(q) / qdot

    such as that

    lmd = R qdot, lmdd = R qddot + Rdot qdot

    tau = -R^T f_m

    Notes
    -----

    (a) The model can represent a 2-link (used for validation) or 3-link system
    by changing nd = 2/3. Unfortunately, the muscle geometry is defined for the
    3-body case.

    (b) The Newton's 3rd law is applied for each force and torque. For example,
    for the first body: tau_1 - tau_2 is applied.

    (c) For the derivation of the equations of motion we used the Euler-Lagrange
    Method assuming:

    M qddot + C qdot + d V / dq = tau, V: potential energy, C: Coriolis Matrix

    """

    def __init__(self, use_gravity=0, use_coordinate_limits=1, use_viscosity=1):
        """
        """
        self.logger = Logger('ArmModel')
        # n: DoFs, md: muscles
        self.nd = 3
        self.md = 9
        # used for selecting since we use 1-based indexing
        self.s = self.nd + 1
        # used for iterating
        self.dim = list(range(1, self.s))
        # enable/disable gravity in EoM [0/1] (simulation too slow)
        self.use_gravity = use_gravity
        # enable/disable coordinate limits in EoM [0/1]
        self.use_coordinate_limits = use_coordinate_limits
        # enable/disable viscous joints in EoM [0/1]
        self.use_viscosity = use_viscosity
        # true if constants are not substituted
        self.sub_constants = True
        # default model state
        self.state0 = np.array([
            np.deg2rad(45.0),  # q1
            np.deg2rad(45.0),  # q2
            np.deg2rad(45.0),  # q3
            np.deg2rad(0.00),  # u1
            np.deg2rad(0.00),  # u2
            np.deg2rad(0.00)   # u3
        ])
        # reference pose used for calculating the optimal fiber length
        self.reference_pose = np.array([
            np.deg2rad(60.0),  # q1
            np.deg2rad(70.0),  # q2
            np.deg2rad(50.0),  # q3
        ])
        self.logger.debug('Constructing model...')
        # construct model
        self.__construct_symbols()
        self.__construct_kinematics()
        self.__construct_coordinate_limiting_forces()
        self.__construct_kinetics()
        self.__construct_drawables()
        self.__construct_muscle_geometry()
        self.__define_muscle_parameters()
        self.__construct_rhs()

    def __construct_symbols(self):
        """Define the symbols used by the analytical model.

        """
        self.logger.debug('__construct_symbols')
        # define model constants 10 a's and b's are used but indexed from 1-9
        # thus 0 is not used (for correspondence with the paper)
        self.a = sp.Matrix(sp.symbols('a0:10'))
        self.b = sp.Matrix(sp.symbols('b0:10'))
        # segment lengths
        self.L = sp.Matrix(sp.symbols('L0:4'))
        # distance to segment's CoM
        self.Lc = sp.Matrix(sp.symbols('Lc0:4'))
        # body parameters
        # inertia
        self.Iz = sp.Matrix(sp.symbols('Iz0:4'))
        # mass
        self.m = sp.Matrix(sp.symbols('m0:4'))
        # time
        self.t = sp.symbols('t')
        # gravity
        self.g = sp.symbols('g')
        # q's are used instead of $\theta$
        self.q = sp.Matrix(dynamicsymbols('theta0:4'))
        self.u = sp.Matrix(dynamicsymbols('u:4'))
        self.dq = sp.Matrix([sp.diff(x, self.t) for x in self.q])
        self.ddq = sp.Matrix([sp.diff(x, self.t) for x in self.dq])
        # tau acting forces
        self.tau = sp.Matrix(dynamicsymbols('tau0:4'))
        # define a dictionary that maps symbols to values
        # parameters are derived from [1]
        self.constants = dict({self.a[1]: 0.055, self.a[2]: 0.055, self.a[3]:
                               0.220, self.a[4]: 0.24, self.a[5]: 0.040,
                               self.a[6]: 0.040, self.a[7]: 0.220, self.a[8]:
                               0.06, self.a[9]: 0.26, self.b[1]: 0.080,
                               self.b[2]: 0.11, self.b[3]: 0.030, self.b[4]:
                               0.03, self.b[5]: 0.045, self.b[6]: 0.045,
                               self.b[7]: 0.048, self.b[8]: 0.050, self.b[9]:
                               0.03, self.L[1]: 0.310, self.L[2]: 0.270,
                               self.L[3]: 0.150, self.Lc[1]: 0.165, self.Lc[2]:
                               0.135, self.Lc[3]: 0.075, self.m[1]: 1.93,
                               self.m[2]: 1.32, self.m[3]: 0.35, self.Iz[1]:
                               0.0141, self.Iz[2]: 0.0120, self.Iz[3]: 0.001,
                               self.g: 9.81})

        # pickle workaround
        for q in self.q:
            q.__class__.__module__ = '__main__'

        for dq in self.dq:
            dq.__class__.__module__ = '__main__'

        for ddq in self.ddq:
            ddq.__class__.__module__ = '__main__'

        for u in self.u:
            u.__class__.__module__ = '__main__'

        for tau in self.tau:
            tau.__class__.__module__ = '__main__'

        # max isometrix force
        # TODO all muscles are equally strong
        fmax = 50
        self.Fmax = sp.diag(fmax, fmax, fmax,
                            fmax, 0.5 * fmax, 0.5 * fmax,
                            fmax, fmax, 0.5 * fmax)

    def __construct_kinematics(self):
        """Define points of interest for the derivation of the EoM. These are used for
        constructing the EoM.

        """
        self.logger.debug('__construct_kinematics')
        L = self.L
        Lc = self.Lc
        q = self.q
        # define the spatial coordinates for the Lc in terms of Lc s' and q's
        # arm
        xc1 = sp.Matrix([Lc[1] * cos(q[1]),
                         Lc[1] * sin(q[1]),
                         0,
                         0,
                         0,
                         q[1]])
        # forearm
        xc2 = sp.Matrix([L[1] * cos(q[1]) + Lc[2] * cos(q[1] + q[2]),
                         L[1] * sin(q[1]) + Lc[2] * sin(q[1] + q[2]),
                         0,
                         0,
                         0,
                         q[1] + q[2]])

        # hand
        xc3 = sp.Matrix([L[1] * cos(q[1]) + L[2] * cos(q[1] + q[2]) +
                         Lc[3] * cos(q[1] + q[2] + q[3]),
                         L[1] * sin(q[1]) + L[2] * sin(q[1] + q[2]) +
                         Lc[3] * sin(q[1] + q[2] + q[3]),
                         0,
                         0,
                         0,
                         q[1] + q[2] + q[3]])
        self.xc = [sp.Matrix([0]), xc1, xc2, xc3]
        # CoM velocities
        self.vc = [sp.diff(x, self.t) for x in self.xc]
        # calculate CoM Jacobian
        self.Jc = [x.jacobian(self.QDot()) for x in self.vc]

    def __construct_kinetics(self):
        """Construct model's dynamics (M, tau_c, tau_g).

        """
        self.logger.debug('__construct_kinetics')
        # generate the mass matrix [6 x 6] for each body
        self.M = [sp.diag(self.m[i], self.m[i], self.m[i], 0, 0,
                          self.Iz[i]) for i in self.dim]
        # dummy 0 for 1-based indexing
        self.M.insert(0, 0)
        # map spatial to generalized inertia
        self.M = [self.Jc[i].T * self.M[i] * self.Jc[i]
                  for i in self.dim]
        # sum the mass product of each body
        self.M = reduce(lambda x, y: x + y, self.M)
        self.M = sp.trigsimp(self.M)
        # Coriolis matrix
        self.C = sp.trigsimp(coriolis_matrix(self.M, self.Q(), self.QDot()))
        # Coriolis forces
        self.tau_c = sp.trigsimp(self.C * sp.Matrix(self.QDot()))
        # potential energy due to gravity force
        self.V = 0
        for i in self.dim:
            self.V = self.V + self.m[i] * self.g * self.xc[i][1]

        self.tau_g = sp.Matrix([sp.diff(self.V, x) for x in self.Q()])

    def __construct_coordinate_limiting_forces(self):
        """Construct coordinate limiting forces.

        """
        self.logger.debug('__construct_coordinate_limiting_forces')

        a = 5
        b = 50
        q_low = [None, np.deg2rad(5), np.deg2rad(5), np.deg2rad(5)]
        q_up = [None, np.deg2rad(175), np.pi, np.deg2rad(100)]

        self.__tau_l = [coordinate_limiting_force(self.q[i], q_low[i], q_up[i], a,
                                                  b) for i in range(1, self.s)]

    def __construct_drawables(self):
        """Construct points of interest (e.g. muscle insertion, CoM, joint centers).

        """
        self.logger.debug('__construct_drawables')
        a = self.a
        b = self.b
        q = self.q
        L = self.L
        # define muscle a
        self.ap = [[-a[1], sp.Rational(0)],  # a1
                   [a[2], sp.Rational(0)],  # a2
                   [a[3] * cos(q[1]), a[3] * sin(q[1])],  # a3
                   [a[4] * cos(q[1]), a[4] * sin(q[1])],  # a4
                   [-a[5], sp.Rational(0)],  # a5
                   [a[6], sp.Rational(0)],  # a6
                   [L[1] * cos(q[1]) + a[7] * cos(q[1] + q[2]),
                    L[1] * sin(q[1]) + a[7] * sin(q[1] + q[2])],  # a7
                   [L[1] * cos(q[1]) + a[8] * cos(q[1] + q[2]),
                    L[1] * sin(q[1]) + a[8] * sin(q[1] + q[2])],  # a8
                   [a[9] * cos(q[1]), a[9] * sin(q[1])]  # a9
                   ]
        # define muscle b
        self.bp = [[b[1] * cos(q[1]), b[1] * sin(q[1])],  # b1
                   [b[2] * cos(q[1]), b[2] * sin(q[1])],  # b2
                   [L[1] * cos(q[1]) + b[3] * cos(q[1] + q[2]),
                    L[1] * sin(q[1]) + b[3] * sin(q[1] + q[2])],  # b3
                   [L[1] * cos(q[1]) - b[4] * cos(q[1] + q[2]),
                    L[1] * sin(q[1]) - b[4] * sin(q[1] + q[2])],  # b4
                   [L[1] * cos(q[1]) + b[5] * cos(q[1] + q[2]),
                    L[1] * sin(q[1]) + b[5] * sin(q[1] + q[2])],  # b5
                   [L[1] * cos(q[1]) - b[6] * cos(q[1] + q[2]),
                    L[1] * sin(q[1]) - b[6] * sin(q[1] + q[2])],  # b6
                   [L[1] * cos(q[1]) + L[2] * cos(q[1] + q[2]) + b[7] * cos(q[1] + q[2] + q[3]),
                    L[1] * sin(q[1]) + L[2] * sin(q[1] + q[2]) + b[7] * sin(q[1] + q[2] + q[3])],  # b7
                   [L[1] * cos(q[1]) + L[2] * cos(q[1] + q[2]) - b[8] * cos(q[1] + q[2] + q[3]),
                    L[1] * sin(q[1]) + L[2] * sin(q[1] + q[2]) - b[8] * sin(q[1] + q[2] + q[3])],  # b8
                   [L[1] * cos(q[1]) + L[2] * cos(q[1] + q[2]) + b[9] * cos(q[1] + q[2] + q[3]),
                    L[1] * sin(q[1]) + L[2] * sin(q[1] + q[2]) + b[9] * sin(q[1] + q[2] + q[3])]  # b9
                   ]
        # define CoM
        self.bc = [[self.xc[1][0], self.xc[1][1]],
                   [self.xc[2][0], self.xc[2][1]],
                   [self.xc[3][0], self.xc[3][1]]
                   ]
        # joint center
        self.jc = [[sp.Rational(0), sp.Rational(0)],
                   [L[1] * cos(q[1]), L[1] * sin(q[1])],
                   [L[1] * cos(q[1]) + L[2] * cos(q[1] + q[2]),
                    L[1] * sin(q[1]) + L[2] * sin(q[1] + q[2])]
                   ]
        # end effector
        if self.nd == 3:
            self.ee = sp.Matrix([L[1] * cos(q[1]) + L[2] * cos(q[1] + q[2]) +
                                 L[3] * cos(q[1] + q[2] + q[3]), L[1] *
                                 sin(q[1]) + L[2] * sin(q[1] + q[2]) + L[3] *
                                 sin(q[1] + q[2] + q[3])])
        elif self.nd == 2:
            self.ee = sp.Matrix([L[1] * cos(q[1]) + L[2] * cos(q[1] + q[2]),
                                 L[1] * sin(q[1]) + L[2] * sin(q[1] + q[2])
                                 ])

    def __construct_muscle_geometry(self):
        """Construct muscle length function and moment arm.

        """
        self.logger.debug('__construct_geometry')
        a = self.a
        b = self.b
        L = self.L
        q = self.q
        # muscle length ($l(q)$)
        self.lm = sp.Matrix([
            (a[1] ** 2 + b[1] ** 2 + 2 * a[1] *
             b[1] * cos(q[1])) ** sp.Rational(1, 2),
            (a[2] ** 2 + b[2] ** 2 - 2 * a[2] *
             b[2] * cos(q[1])) ** sp.Rational(1, 2),
            ((L[1] - a[3]) ** 2 + b[3] ** 2 + 2 * (L[1] - a[3])
             * b[3] * cos(q[2])) ** sp.Rational(1, 2),
            ((L[1] - a[4]) ** 2 + b[4] ** 2 - 2 * (L[1] - a[4])
             * b[4] * cos(q[2])) ** sp.Rational(1, 2),
            (a[5] ** 2 + b[5] ** 2 + L[1] ** 2 + 2 * a[5] * L[1] * cos(q[1]) +
             2 * b[5] * L[1] * cos(q[2]) + 2 * a[5] * b[5] * cos(q[1] + q[2])) ** sp.Rational(1, 2),
            (a[6] ** 2 + b[6] ** 2 + L[1] ** 2 - 2 * a[6] * L[1] * cos(q[1]) -
             2 * b[6] * L[1] * cos(q[2]) + 2 * a[6] * b[6] * cos(q[1] + q[2])) ** sp.Rational(1, 2),
            ((L[2] - a[7]) ** 2 + b[7] ** 2 + 2 * (L[2] - a[7])
             * b[7] * cos(q[3])) ** sp.Rational(1, 2),
            ((L[2] - a[8]) ** 2 + b[8] ** 2 - 2 * (L[2] - a[8])
             * b[8] * cos(q[3])) ** sp.Rational(1, 2),
            ((L[1] - a[9]) ** 2 + b[9] ** 2 + L[2] ** 2 + 2 * (L[1] - a[9]) * L[2] * cos(q[2]) +
             2 * b[9] * L[2] * cos(q[3]) +
             2 * (L[1] - a[9]) * b[9] * cos(q[2] + q[3])) ** sp.Rational(1, 2)
        ])

        self.lmd = sp.diff(self.lm, self.t)
        self.lmdd = sp.diff(self.lmd, self.t)

        # calculate the moment arm matrix and its derivatives
        self.R = sp.trigsimp(self.lm.jacobian(self.Q()))
        self.RDot = sp.diff(self.R, self.t)
        self.RDotQDot = self.RDot * sp.Matrix(self.U())

    def __define_muscle_parameters(self):
        """Define muscle parameters, such as optimal fiber length (reference pose) and
        tendon stiffness.

        """
        self.logger.debug('__define_muscle_parameters')
        parameters = self.model_parameters(q=self.reference_pose,
                                           in_deg=False)
        self.lm0 = self.lm.subs(parameters)
        # the derivative of a matrix with a vector is a rank 3 tensor (3D
        # array), [dM/dq1, dM/dq2, ...]
        self.RTDq = sp.derive_by_array(self.R.transpose(), self.Q())

    def __construct_rhs(self):
        """Construct a callable function that can be used to integrate the mode.

        rhs = rhs(x, t, controller specifieds, parameters values)

        """
        self.logger.debug('__construct_rhs')
        self.logger.debug('Use Gravity: ' + str(self.use_gravity))
        self.logger.debug('Use Coordinate Limits: ' +
                          str(self.use_coordinate_limits))
        self.logger.debug('Use Viscous Joints: ' + str(self.use_viscosity))

        # forces
        # Newton's 3rd law
        b = 0.05
        tau = sp.Matrix(apply_generalized_force(self.Tau()))
        self.tau_l = sp.Matrix(apply_generalized_force(self.__tau_l))
        self.tau_b = -b * sp.Matrix(apply_generalized_force(self.U()))
        # tau = sp.Matrix(self.Tau())
        # self.tau_l = sp.Matrix(self.__tau_l)
        # self.tau_b = -b *sp.Matrix(self.U())

        # M qdd + tau_c + tau_g = tau + tau_l + tau_b-> M qdd = forcing
        # f = tau_c + tau_g - tau_l - tau_b
        self.f = self.tau_c \
            + self.use_gravity * self.tau_g \
            - self.use_coordinate_limits * self.tau_l \
            - self.use_viscosity * self.tau_b
        self.forcing = tau - self.f

        # substitute dq with u (required for code-gen)
        for i in range(0, self.forcing.shape[0]):
            self.forcing = self.forcing.subs(self.dq[i + 1], self.u[i + 1])

        # rhs
        self.coordinates = sp.Matrix(self.Q())
        self.speeds = sp.Matrix(self.U())
        self.coordinates_derivatives = self.speeds
        self.specifieds = sp.Matrix(self.Tau())
        self.rhs = generate_ode_function(
            self.forcing,
            self.coordinates,
            self.speeds,
            list(self.constants.keys()),
            mass_matrix=self.M,
            coordinate_derivatives=self.coordinates_derivatives,
            specifieds=self.specifieds)

# ------------------------------------------------------------------------
# ArmModel public interface
# ------------------------------------------------------------------------

    def Q(self):
        'Get active coordinates (q) [1, s].'
        return self.q[1:self.s]

    def QDot(self):
        'Get active speeds (qdot) [1, s].'
        return self.dq[1:self.s]

    def U(self):
        'Get active speeds (qdot = u) [1, s].'
        return self.u[1:self.s]

    def Tau(self):
        'Get active speeds (tau) [1, s].'
        return self.tau[1:self.s]

    def model_parameters(self, **kwargs):
        """Get the model parameters dictionary given q, u, in_deg=[True/False].

        Parameters
        ----------

        kwargs: q=q, u=u, in_deg=[True/False]

        """

        expected_args = ["q", "u", "in_deg"]
        kwargsdict = {}
        for key in list(kwargs.keys()):
            if key in expected_args:
                kwargsdict[key] = kwargs[key]
            else:
                raise Exception("Unexpected Argument")

        return self.__model_parameters(kwargsdict)

    def __model_parameters(self, dic):
        """A private implementation of model_parameters.

        Parameters
        ----------

        dic: dictionary containing {q, u, in_deg}

        """

        constants = {}
        if self.sub_constants:
            constants = self.constants.copy()

        q = self.Q()
        dq = self.QDot()
        u = self.U()
        in_deg = False
        if "in_deg" in list(dic.keys()):
            in_deg = dic["in_deg"]

        if "q" in list(dic.keys()):
            qs = dic["q"]
            if in_deg:
                qs = np.deg2rad(dic["q"])

            constants.update({q[i]: qs[i] for i in range(0, self.nd)})

        if "u" in list(dic.keys()):
            us = dic["u"]

            constants.update({dq[i]: us[i] for i in range(0, self.nd)})

            constants.update({u[i]: us[i] for i in range(0, self.nd)})

        return constants

    def pre_substitute_parameters(self):
        """Substitute model parameters into the variables of the model to improve speed.

        """

        self.logger.debug('pre_substitute_parameters')
        self.sub_constants = False
        constants = self.constants

        self.M = self.M.subs(constants)
        self.tau_c = self.tau_c.subs(constants)
        self.tau_g = self.tau_g.subs(constants)
        self.tau_l = self.tau_l.subs(constants)
        self.tau_b = self.tau_b.subs(constants)
        self.f = self.f.subs(constants)

        self.lm = self.lm.subs(constants)
        self.lmd = self.lmd.subs(constants)
        self.lmdd = self.lmdd.subs(constants)
        self.R = self.R.subs(constants)
        self.RDot = self.RDot.subs(constants)
        self.RDotQDot = self.RDotQDot.subs(constants)
        self.RTDq = self.RTDq.subs(constants)

        self.ap = substitute(self.ap, constants)
        self.bp = substitute(self.bp, constants)
        self.bc = substitute(self.bc, constants)
        self.jc = substitute(self.jc, constants)
        self.ee = substitute(self.ee, constants)

        # self.L = self.L.subs(constants)
        # self.Lc = self.Lc.subs(constants)
        # self.Iz = self.Iz.subs(constants)

        self.xc = substitute(self.xc, constants)
        self.vc = substitute(self.vc, constants)
        self.Jc = substitute(self.Jc, constants)

        # self.m = self.m.subs(constants)
        # self.g = self.g.subs(constants)

    def draw_model(self, q, in_deg, ax=None, scale=0.8, use_axis_limits=True, alpha=1.0,
                   text=True):
        """Draws the 3D toy model.

        Parameters
        ----------

        q: coordinate values in degrees or rad
        in_deg: True/False
        ax: axis 1D
        scale: if figure is small this helps in visualizing details
        use_axis_limits: use axis limits from max length

        """
        if self.nd == 2:
            self.logger.debug('draw_model supports 3DoFs case')
            return

        constants = self.model_parameters(q=q, in_deg=in_deg)

        joints = substitute(self.jc, constants)
        muscle_a = substitute(self.ap, constants)
        muscle_b = substitute(self.bp, constants)
        end_effector = substitute(self.ee, constants)
        CoM = substitute(self.bc, constants)

        linewidth = 4 * scale
        gd_markersize = 14 * scale
        jc_markersize = 12 * scale
        mo_markersize = 7 * scale
        ef_markersize = 15 * scale
        fontsize = 12 * scale

        if ax == None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        # arm
        ax.plot([joints[0, 0], joints[1, 0]], [joints[0, 1], joints[1, 1]],
                'r', linewidth=linewidth, alpha=alpha)
        # forearm
        ax.plot([muscle_b[5, 0], joints[2, 0]], [muscle_b[5, 1], joints[2, 1]],
                'r', linewidth=linewidth, alpha=alpha)
        # hand
        ax.plot([muscle_b[7, 0], end_effector[0]], [muscle_b[7, 1], end_effector[1]],
                'r', linewidth=linewidth, alpha=alpha)
        # muscles
        for i in range(0, muscle_a.shape[0]):
            ax.plot([muscle_a[i, 0], muscle_b[i, 0]], [
                muscle_a[i, 1], muscle_b[i, 1]], 'b', alpha=alpha)
            if text:
                ax.text(muscle_a[i, 0], muscle_a[i, 1],
                        r'$a_' + str(i + 1) + '$', fontsize=fontsize, alpha=alpha)
                ax.text(muscle_b[i, 0], muscle_b[i, 1],
                        r'$b_' + str(i + 1) + '$', fontsize=fontsize, alpha=alpha)
                ax.text((muscle_b[i, 0] + muscle_a[i, 0]) / 2.0,
                        (muscle_b[i, 1] + muscle_a[i, 1]) / 2.0,
                        r'$l_' + str(i + 1) + '$', fontsize=fontsize, alpha=alpha)

        # joint centers
        ax.plot(joints[:, 0], joints[:, 1], 'or',
                markersize=gd_markersize, alpha=alpha)
        if text:
            for i in range(0, joints.shape[0]):
                ax.text(joints[i, 0], joints[i, 1], r'$J_' +
                        str(i + 1) + '$', fontsize=fontsize, alpha=alpha)

        # CoM
        ax.plot(CoM[:, 0], CoM[:, 1], 'oy',
                markersize=jc_markersize, alpha=alpha)
        if text:
            for i in range(0, CoM.shape[0]):
                ax.text(CoM[i, 0], CoM[i, 1], r'$Lc_' +
                        str(i + 1) + '$', fontsize=fontsize, alpha=alpha)

        # end effector
        ax.plot(end_effector[0], end_effector[1],
                '<b', markersize=ef_markersize, alpha=alpha)
        # muscle origin
        ax.plot(muscle_a[:, 0], muscle_a[:, 1], 'dy',
                markersize=mo_markersize, alpha=alpha)
        # muscle insertion
        ax.plot(muscle_b[:, 0], muscle_b[:, 1], 'db',
                markersize=mo_markersize, alpha=alpha)

        ax.axis('equal')
        ax.set_title('Model Pose')
        ax.set_xlabel('$x \; (m)$')
        ax.set_ylabel('$y \; (m)$')

        # axis limits
        if use_axis_limits:
            L_max = self.constants[self.L[1]] + \
                self.constants[self.L[2]] + self.constants[self.L[3]]
            ax.set_xlim([-L_max, L_max])
            ax.set_ylim([-L_max / 2, 1.5 * L_max])


# ------------------------------------------------------------------------
# ArmModel tests
# ------------------------------------------------------------------------

    def test_muscle_geometry(self):
        """Evaluate the muscle geometry for a random pose against the ground truth.

        """
        ma = self.ap
        mb = self.bp
        lmt = sp.Matrix([sp.trigsimp(
            sp.sqrt(pow(ma[i][0] - mb[i][0], 2) +
                    pow(ma[i][1] - mb[i][1], 2))) for i in range(0, len(ma))])

        pose = self.model_parameters(q=np.random.random(3), in_deg=False)
        assert_if_same(self.lm.subs(pose), lmt.subs(pose))
