# This script calculates the feasible muscle forces that satisfy the action
# (motion) and the physiological constraints of the muscles. The requirements
# for the calculation are the model, the motion from inverse kinematics, the
# muscle forces from static optimization. Results must be stored in the
# appropriate directory so that perform_joint_reaction_batch.py can locate the
# files.
#
# @author Dimitar Stanev (jimstanev@gmail.com)
#
import os
import pickle
import numpy as np
from tqdm import tqdm
from opensim_utils import calculate_muscle_data
from util import null_space, construct_muscle_space_inequality, \
    convex_bounded_vertex_enumeration,  readMotionFile, \
    index_containing_substring, write_as_sto_file

###############################################################################
# parameters

# when computed once results are stored into files so that they can be loaded
# with (pickle)
compute = True
subject_dir = os.getcwd() + '/../dataset/Gait10dof18musc/'
model_file = subject_dir + 'subject01.osim'
ik_file = subject_dir + 'notebook_results/subject01_walk_ik.mot'
so_file = subject_dir + 'notebook_results/subject01_walk_StaticOptimization_force.sto'
feasible_set_dir = subject_dir + 'notebook_results/feasible_force_set/'

# read opensim files
if not (os.path.isfile(model_file) and
        os.path.isfile(ik_file) and
        os.path.isfile(so_file)):
    raise RuntimeError('required files do not exist')

if not os.path.isdir(feasible_set_dir):
    raise RuntimeError('required folders do not exist')

###############################################################################

if compute:
    moment_arm, max_force = calculate_muscle_data(model_file, ik_file)

    so_header, so_labels, so_data = readMotionFile(so_file)
    so_data = np.array(so_data)

    coordinates = moment_arm[0].shape[0]
    muscles = moment_arm[0].shape[1]
    time = so_data[:, 0]
    entries = time.shape[0]

    # collect quantities for computing the feasible muscle forces
    NR = []
    Z = []
    b = []
    fm_par = []
    print('Collecting data ...')
    for t in tqdm(range(0, entries)):
        # get tau, R, Fmax
        fm = so_data[t, 1:(muscles + 1)]  # time the first column
        RT_temp = moment_arm[t, :, :]
        fmax_temp = max_force[t, :]

        # calculate the reduced rank (independent columns) null space to avoid
        # singularities
        NR_temp = null_space(RT_temp)
        # fm_par = fm is used instead of fm_par = -RBarT * tau because the
        # muscle may not be able to satisfy the action. In OpenSim residual
        # actuators are used to ensure that Static Optimization can satisfy the
        # action. In this case, we ignore the reserve forces and assume that fm
        # is the minimum effort solution. If the model is able to satisfy the
        # action without needing reserve forces then we can use fm_par = -RBarT
        # * tau as obtained form Inverse Dynamics.
        # A better implementation that usese fm_par = -RBarT * tau is provided:
        # https://github.com/mitkof6/feasible_muscle_force_analysis
        # this implementation also supports nonlinear muscles and can excluded 
        # muscles and coordinates from the analysis
        fm_par_temp = fm

        Z_temp, b_temp = construct_muscle_space_inequality(NR_temp,
                                                           fm_par_temp,
                                                           fmax_temp)

        # append results
        NR.append(NR_temp)
        Z.append(Z_temp)
        b.append(b_temp)
        fm_par.append(fm_par_temp)

    # calculate the feasible muscle force set
    print('Calculating null space ...')
    f_set = []
    for t in tqdm(range(0, entries)):
        try:
            fs = convex_bounded_vertex_enumeration(Z[t], b[t][:, 0], 0,
                                                   method='lrs')
        except:
            print('inequlity is infeasible thus append previous iteration')
            f_set.append(f_set[-1])
            continue

        temp = []
        for i in range(0, fs.shape[0]):
            temp.append(fm_par[t] + NR[t].dot(fs[i, :]))

        f_set.append(temp)

    # serialization f_set -> [time x feasible force set set x muscles]
    pickle.dump(f_set, file(subject_dir + 'f_set.dat', 'w'))

else:  # store the feasible set into multiple .sto files
    # load data
    f_set = pickle.load(file(subject_dir + 'f_set.dat', 'r'))
    so_header, so_labels, so_data = readMotionFile(so_file)
    so_data = np.array(so_data)
    time = so_data[:, 0]
    # keep only muscle forces
    idx = index_containing_substring(so_labels, 'FX')[0]
    labels = so_labels[:idx]

    # find largest feasible set
    S = len(f_set[0])
    for fs in f_set:
        if len(fs) > S:
            S = len(fs)

    # export muscle force realizations
    print('Exporting feasible force set as .sto ...')
    for j in tqdm(range(0, S)):
        data_temp = []
        for i in range(0, so_data.shape[0]):
            data_temp.append(f_set[i][j % len(f_set[i])])

        write_as_sto_file(feasible_set_dir + str(j) + '.sto', labels,
                          time, np.array(data_temp))
