import numpy as np
import sympy as sp
from logger import Logger
from scipy.optimize import minimize
from util import to_np_mat, to_np_vec, sigmoid, rotate, gaussian
from simulation import SimulationReporter
from delay import DelayArray


# ------------------------------------------------------------------------
# PID
# ------------------------------------------------------------------------


class PD:
    """Proportional Derivative Controller.

    a = ad + Kp (xd - x) + Kd (ud - u)

    """

    def __init__(self, Kp, Kd):
        """
        Parameters
        ----------

        Kp: proportional gain

        Kd: derivative gain

        """

        self.Kp = Kp
        self.Kd = Kd
        self.t = 0

    def compute(self, x, u, xd, ud, ad):
        """
        Parameters
        ----------

        x: current position
        u: current velocity
        xd: desired position
        ud: desired velocity
        ad: desired acceleration

        Returns
        -------

        the requred acceleration

        """

        return ad + self.Kp * (xd - x) + self.Kd * (ud - u)


class PID:
    """
    An implementation of Proportional Integral Derivative controller.

    e = xd - x
    u = Kp e + Ki \int_0^t e d\tau + Kd de/dt

    """

    def __init__(self, Kp, Ki, Kd):
        """
        Parameters
        ----------

        Kp: proportional gain
        Ki: integral gain
        Kd: derivative gain

        """

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.t = 0
        self.error_sum = 0

    def compute(self, t, x, u, xd, ud):
        """
        Parameters
        ----------

        x: current position
        u: current velocity
        xd: desired position
        ud: desired velocity

        Returns
        -------

        the requred acceleration

        Note: TODO if the numerical integration goes backward in time then this
        may cause instabilities

        """

        dt = np.abs(self.t - t)

        error = xd - x
        error_d = ud - u
        self.error_sum = self.error_sum + error * dt
        self.t = t

        return self.Kp * error + self.Kd * error_d + self.Ki * self.error_sum


# ------------------------------------------------------------------------
# JointSpaceController
# ------------------------------------------------------------------------


class JointSpaceController:
    """Simple joint space tracking controller.

    """

    def __init__(self, model):
        """ Constructor.

        Parameters
        ----------

        model: reference to the model

        """
        self.logger = Logger('JointSpaceController')
        self.model = model
        self.pid = PID(10, 2, 2)
        self.reporter = SimulationReporter(model)

    def controller(self, x, t, x0):
        """Default simple joint space PD controller.

        Parameters
        ----------

        x: state
        t: time
        x0: initial state

        """
        # self.logger.debug('Time: ' + str(t))
        n = self.model.nd
        q = np.array(x[:n])
        u = np.array(x[n:])
        qd = np.array(self.target(x, t, x0))
        ud = np.zeros((n))
        tau = np.array(self.pid.compute(t, q, u, qd, ud))

        # record
        self.reporter.t.append(t)
        self.reporter.q.append(q)
        self.reporter.qd.append(qd)
        self.reporter.tau.append(tau)
        # self.fs_reporter.record(t, q, u, tau.reshape((n, 1)))

        return tau

    def target(self, x, t, x0):
        """Default joint space target function.

        Parameters
        ----------

        x: state
        t: time
        x0: initial state

        """
        return [np.deg2rad(130), np.deg2rad(60), np.deg2rad(95)]

# ------------------------------------------------------------------------
# TaskSpaceController
# ------------------------------------------------------------------------


class TaskSpaceController:
    """A task space controller that moves the task goal in a direction.

    """

    def __init__(self, model, task, angle=0, evaluate_muscle_forces=False):
        """Constructor.

        Parameters
        ----------

        model: Model
            a reference to the model

        task: TaskSpace
            a reference to TaskSpace

        angle: rad (default=0)
            direction to move the task

        evaluate_muscle_forces: Boolean (default=False)
            compute muscle forces that satisfy the task constraint

        """
        self.logger = Logger('TaskSpaceController')
        self.model = model
        self.task = task
        self.angle = angle
        self.evaluate_muscle_forces = evaluate_muscle_forces
        self.pd = PD(50, 5)
        self.reporter = SimulationReporter(model)

    def controller(self, x, t, x0):
        """Controller function.

        Parameters
        ----------

        x: state
        t: time
        x0: initial state

        """
        self.logger.debug('Time: ' + str(t))
        n = self.model.nd
        q = np.array(x[:n])
        u = np.array(x[n:])
        pose = self.model.model_parameters(q=q, u=u)

        # task variables
        xc = to_np_mat(self.task.x(pose))
        uc = to_np_mat(self.task.u(pose, u))
        xd, ud, ad = self.target(x, t, x0)
        ad = sp.Matrix(self.pd.compute(xc, uc, xd, ud, ad))

        # forces
        tau, ft = self.task.calculate_force(ad, pose)
        ft = to_np_vec(ft)
        tau = to_np_vec(tau)

        # solve static optimization
        fm = None
        if self.evaluate_muscle_forces:
            m = self.model.md
            R = to_np_mat(self.model.R.subs(pose))
            RT = R.transpose()

            def objective(x):
                return np.sum(x**2)

            def inequality_constraint(x):
                return np.array(tau + RT * (x.reshape(-1, 1))).reshape(-1,)

            x0 = np.zeros(m)
            bounds = tuple([(0, self.model.Fmax[i, i]) for i in range(0, m)])
            constraints = ({'type': 'ineq', 'fun': inequality_constraint})
            sol = minimize(objective, x0,  method='SLSQP',
                           bounds=bounds,
                           constraints=constraints)
            if sol.success is False:
                raise Exception('Static optimization failed at: ' + t)

            fm = sol.x.reshape(-1,)

        # record
        self.reporter.t.append(t)
        self.reporter.q.append(q)
        self.reporter.u.append(u)
        self.reporter.x.append(np.array(xc).reshape(-1,))
        self.reporter.xd.append(np.array(xd).reshape(-1,))
        self.reporter.tau.append(tau)
        self.reporter.ft.append(ft)
        self.reporter.fm.append(fm)
        # self.fs_reporter.record(t, q, u, tau)

        return tau

    def target(self, x, t, x0):
        """ A directed sigmoid function target.

        Parameters
        ----------

        x: state
        t: time
        x0: initial state

        Returns
        -------

        (x, u, a)

        """
        pose0 = self.model.model_parameters(q=x0[:self.model.nd])
        xt0 = self.task.x(pose0)

        t0 = 1
        A = 0.3
        B = 4
        o = np.asmatrix([[0], [0]])
        xd, ud, ad = sigmoid(t, t0, A, B)

        xd = rotate(o, np.asmatrix([[xd], [0]]), self.angle)
        ud = rotate(o, np.asmatrix([[ud], [0]]), self.angle)
        ad = rotate(o, np.asmatrix([[ad], [0]]), self.angle)

        return (np.asmatrix(xt0 + xd),
                np.asmatrix(ud),
                np.asmatrix(ad))


# ------------------------------------------------------------------------
# MuscleSpaceControllerJS
# ------------------------------------------------------------------------


class MuscleSpaceControllerJS:
    """This controller uses the muscle space EoM to driving a model from a
    reference pose to a desired pose in joint space.

    """

    def __init__(self, model, musclespace):
        """Constructor.

        Parameters
        ----------

        model: a reference to the model
        musclespace: a reference to MuscleSpace

        """
        self.logger = Logger('MsucleSpaceControllerJS')
        self.model = model
        self.musclespace = musclespace
        self.pid = PID(10, 0, 10)
        self.reporter = SimulationReporter(model)

    def controller(self, x, t, x0):
        """Default simple joint space PD controller.

        Parameters
        ----------

        x: state
        t: time
        x0: initial state

        """
        self.logger.debug('Time: ' + str(t))
        n = self.model.nd
        m = self.model.md
        q = np.array(x[:n])
        u = np.array(x[n:])
        qd = np.array(self.target(x, t, x0))
        ud = np.zeros((n))
        qddot = np.array(self.pid.compute(t, q, u, qd, ud))

        # evalutate desired lmdd_d
        pose = self.model.model_parameters(q=q, u=u)
        RDotQDot = self.model.RDotQDot.subs(pose)
        RQDDot = self.model.R.subs(pose) * sp.Matrix(qddot)
        lmdd_d = sp.Matrix(RDotQDot + RQDDot)
        lmd = to_np_vec(self.model.lmd.subs(pose))
        lmd_d = to_np_vec(self.model.R.subs(pose) * sp.Matrix(u))

        tau = self.musclespace.calculate_force(lmdd_d, pose)[0]
        tau = to_np_vec(tau)

        # store record
        self.reporter.t.append(t)
        self.reporter.q.append(q)
        self.reporter.u.append(u)
        self.reporter.qd.append(qd)
        self.reporter.lmd.append(lmd)
        self.reporter.lmd_d.append(lmd_d)
        self.reporter.tau.append(tau)

        return tau

    def target(self, x, t, x0):
        """Default joint space target function.

        Parameters
        ----------

        x: state
        t: time
        x0: initial state

        """
        return [np.deg2rad(130), np.deg2rad(60), np.deg2rad(95)]


# ------------------------------------------------------------------------
# PosturalMuscleSpaceController
# ------------------------------------------------------------------------


class PosturalMuscleSpaceController:
    """This is a posture controller in muscle space. The system is disturbed and a
    muscle length controller is responsible for restoring it to its initial
    pose.

    """

    def __init__(self, model, musclespace, kp, kd, delay, a, t0, sigma, gamma):
        """ Constructor.

        Parameters
        ----------

        model: a reference to ToyModel

        muscle: a reference to TaskSpace

        kp: proportional gain

        kd: derivative gain

        delay: the delay of the reflex loops

        a: Gaussian aptitude (disturbance)

        t0: time of application (disturbance)

        sigma: outspread (disturbance)

        gamma: direction of disturbance

        """
        self.logger = Logger('PosturalMsucleSpaceController')
        self.model = model
        self.musclespace = musclespace
        self.pd = PD(kp, kd)  # (10, 10) -> full controller, (0, 10) -> reflex
        self.delay = delay
        self.a = a
        self.t0 = t0
        self.sigma = sigma
        self.gamma = gamma
        self.reporter = SimulationReporter(model)
        self.__initialize_delay_components()
        self.__calculate_task()

    def __calculate_task(self):
        """Calculate model's end effector Jacobian transpose.

        """
        xt = sp.Matrix(self.model.ee)
        Jt = xt.jacobian(self.model.Q())
        self.JtT = Jt.transpose()

    def __add_in_distrurbance(self, t, tau, pose):
        """Adds distrubance to end effector.

        """
        angle = self.gamma
        o = np.asmatrix([[0], [0]])
        fd = np.asmatrix([[gaussian(t, self.a, self.t0, self.sigma)], [0]])
        fd = rotate(o, fd, angle)
        return tau + self.JtT.subs(pose) * fd

    def __initialize_delay_components(self):
        """Initializes the delay components with the default muscle length (state(0))
        and zero length velocities.

        """
        n = self.model.nd
        m = self.model.md

        state0 = self.model.state0
        q = state0[:n]
        u = state0[n:]
        pose = self.model.model_parameters(q=q, u=u)
        self.lm0 = self.model.lm.subs(pose)
        self.lm0 = to_np_vec(self.lm0)

        delay = np.full(m, self.delay)

        self.lm_del = DelayArray(m, delay, self.lm0)
        self.lmd_del = DelayArray(m, delay, np.zeros(m))

    def controller(self, x, t, x0):
        """ Controller.

        Parameters
        ----------

        x: state
        t: time
        x0: initial state

        """
        self.logger.debug('Time: ' + str(t))
        m = self.model.md
        n = self.model.nd
        q = np.array(x[:n])
        u = np.array(x[n:])

        # compute current muscle length and derivative
        pose = self.model.model_parameters(q=q, u=u)
        lm = self.model.lm.subs(pose)
        lm = to_np_vec(lm)
        lmd = self.model.lmd.subs(pose)
        lmd = to_np_vec(lmd)

        # compute target
        lm_des = self.target(x, t, x0)
        lmd_des = np.zeros(m)
        lmdd_des = np.zeros(m)

        # update delayed values and get current (must update first)
        self.lm_del.add(t, lm)
        self.lmd_del.add(t, lmd)

        lm_del = np.array(self.lm_del.get_delayed())
        lmd_del = np.array(self.lmd_del.get_delayed())

        lmdd_des = sp.Matrix(self.pd.compute(
            lm_del, lmd_del, lm_des, lmd_des, lmdd_des))

        tau, fm = self.musclespace.calculate_force(lmdd_des, pose)
        tau = self.__add_in_distrurbance(t, tau, pose)
        tau = to_np_vec(tau)

        # record
        self.reporter.t.append(t)
        self.reporter.q.append(q)
        self.reporter.u.append(u)
        self.reporter.lm.append(lm)
        self.reporter.lm_d.append(lm_des)
        self.reporter.fm.append(fm)
        self.reporter.tau.append(tau)

        return tau

    def target(self, x, t, x0):
        """Default muscle space target function.

        Parameters
        ----------

        x: state
        t: time
        x0: initial state

        """
        return self.lm0
