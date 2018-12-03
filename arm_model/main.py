import sympy as sp
import numpy as np
import pylab as plt
from model import ArmModel
from simulation import Simulation
from projection import TaskSpace, MuscleSpace
from controller import JointSpaceController, TaskSpaceController,\
    MuscleSpaceControllerJS, PosturalMuscleSpaceController
from analysis import FeasibleMuscleSetAnalysis
from util import calculate_feasible_muscle_set
from sympy import init_printing
from IPython.display import display

import logging
logging.basicConfig(level=logging.DEBUG)

# basic configuration
init_printing(use_unicode=True, wrap_line=False,
              no_global=True, use_latex=True)  # 'mathjax'
# np.set_printoptions(precision=3)
plt.ioff()
# plt.ion()

# ------------------------------------------------------------------------
# utilities
# ------------------------------------------------------------------------


def joint_space_control(model, fig_name='results/js_control', format='pdf'):
    """Make use of the joint space controller to simulate a movement.

    """
    # b = 0.05 produces smooth profiles
    t_end = 5.0
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))

    # controller
    controller = JointSpaceController(model)

    # numerical integration
    simulation = Simulation(model, controller)
    simulation.integrate(t_end)
    simulation.plot_simulation(ax[0])
    controller.reporter.plot_joint_space_data(ax[1])

    fig.tight_layout()
    fig.savefig(fig_name + '.' + format, format=format, dpi=300)
    fig.savefig(fig_name + '.' + 'eps', format='eps', dpi=300)
    fig.show()

    return controller, t_end


def task_space_control(model, angle, evaluate_muscle_forces,
                       fig_name='results/ts_control', format='pdf'):
    """Make use of the task space controller to simulate a movement.

    """
    t_end = 2.0
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))

    # define the end effector position in terms of q's
    end_effector = sp.Matrix(model.ee)
    display('x_t = ', end_effector)

    # task space
    task = TaskSpace(model, end_effector)
    controller = TaskSpaceController(
        model, task, angle, evaluate_muscle_forces)

    # numerical integration
    simulation = Simulation(model, controller)
    simulation.integrate(t_end)
    simulation.plot_simulation(ax[0])
    controller.reporter.plot_task_space_data(ax[1])

    fig.tight_layout()
    fig.savefig(fig_name + '.' + format, format=format, dpi=300)
    fig.savefig(fig_name + '.' + 'eps', format='eps', dpi=300)
    fig.show()

    return controller, t_end, task


def muscle_space_control(model, use_optimization,
                         fig_name='results/ms_control', format='pdf'):
    """Make use of the muscle space controller to simulate a movement.

    """
    # requires small b = 0.01 or 0
    t_end = 3.0
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))

    # muscle space
    muscle_space = MuscleSpace(model, use_optimization)
    controller = MuscleSpaceControllerJS(model, muscle_space)

    # numerical integration
    simulation = Simulation(model, controller)
    simulation.integrate(t_end)
    simulation.plot_simulation(ax[0])
    controller.reporter.plot_muscle_space_data_js(ax[1])

    fig.tight_layout()
    fig.savefig(fig_name + '.' + format, format=format, dpi=300)
    fig.savefig(fig_name + '.' + 'eps', format='eps', dpi=300)
    fig.show()


def postural_muscle_space_control(model, kp, use_optimization,
                                  fig_name='results/pc_control', format='pdf'):
    """Make use of the muscle space controller for posture control.

    """
    t_end = 3.0
    kd = 10
    delay = 0.02
    a = 15.0
    t0 = 0.1
    sigma = 0.01
    gamma = np.pi
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))

    # muscle space
    muscle_space = MuscleSpace(model, use_optimization=use_optimization)
    controller = PosturalMuscleSpaceController(model, muscle_space, kp, kd,
                                               delay, a, t0, sigma, gamma)

    # numerical integration
    simulation = Simulation(model, controller)
    simulation.integrate(t_end)
    simulation.plot_simulation(ax[0])
    controller.reporter.plot_postural_muscle_space_data(ax[1])

    fig.tight_layout()
    fig.savefig(fig_name + '.' + format, format=format, dpi=300)
    fig.savefig(fig_name + '.' + 'eps', format='eps', dpi=300)
    fig.show()


def export_eom(model):
    """Exports equations of motion of the model in a latex format.

    """
    M = model.M
    f = model.f
    R = model.R
    for i in range(0, M.shape[0]):
        for j in range(0, M.shape[1]):
            print('\\begin{dmath}')
            print('M_{' + str(i + 1) + ',' + str(j + 1) + '} = ' +
                  sp.latex(M[i, j], mode='plain'))
            print('\\end{dmath}')

    for i in range(0, f.shape[0]):
        print('\\begin{dmath}')
        print('f_' + str(i + 1) + ' = ' + sp.latex(f[i], mode='plain'))
        print('\\end{dmath}')

    for i in range(0, R.shape[0]):
        for j in range(0, R.shape[1]):
            print('\\begin{dmath}')
            print('R_{' + str(i + 1) + ',' + str(j + 1) + '} = ' +
                  sp.latex(R[i, j], mode='plain'))
            print('\\end{dmath}')


def draw_model(model):
    """Draw a model in a pre-defined pose.

    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), frameon=False)
    model.draw_model([60, 70, 50], True, ax, 1, False)
    fig.tight_layout()
    fig.savefig('results/arm_model.pdf', dpi=600, format='pdf',
                transparent=True, pad_inches=0, bbox_inches='tight')
    plt.show()


# ------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------

plt.close('all')
case_study = 1
if case_study == 0:  # joint space
    model = ArmModel(use_gravity=0, use_coordinate_limits=0, use_viscosity=0)
    model.pre_substitute_parameters()
    base_name = 'results/joint_space_control/joint_space'
    controller, t_end = joint_space_control(model, fig_name=base_name)
elif case_study == 1:  # task space and feasible muscle forces
    model = ArmModel(use_gravity=0, use_coordinate_limits=1, use_viscosity=1)
    model.pre_substitute_parameters()
    base_name = 'results/task_space_control/task_space'
    controller, t_end, task = task_space_control(model, np.pi, False,
                                                 fig_name=base_name)
    feasible_muscle_set = FeasibleMuscleSetAnalysis(model, controller.reporter)
    base_name = 'results/feasible_muscle_forces/feasible_forces_ts180_'
    calculate_feasible_muscle_set(feasible_muscle_set,
                                  base_name, 0.0, t_end,
                                  0.1, 500)
elif case_study == 2:  # posture
    model = ArmModel(use_gravity=0, use_coordinate_limits=0, use_viscosity=0)
    model.pre_substitute_parameters()
    base_name = 'results/posture_control/posture_full'
    postural_muscle_space_control(model, 10, True, fig_name=base_name)
    base_name = 'results/posture_control/posture_reflex'
    postural_muscle_space_control(model, 0, True, fig_name=base_name)
elif case_study == 3:  # muscle space
    model = ArmModel(use_gravity=0, use_coordinate_limits=0, use_viscosity=0)
    model.pre_substitute_parameters()
    base_name = 'results/muscle_space_control/muscle_space'
    muscle_space_control(model, False, fig_name=base_name)
elif case_study == 4:  # model export
    model = ArmModel(use_gravity=1, use_coordinate_limits=0, use_viscosity=0)
    draw_model(model)
    export_eom(model)
else:
    print('Undefined case')
