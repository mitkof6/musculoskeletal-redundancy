# This script calculates the feasible muscle forces that satisfy the action
# (motion) and the physiological constraints of the muscles. The requirements
# for the calculation are the model, the motion from inverse kinematics, the
# generalized forces tau from inverse dynamics.
#
# @author Dimitar Stanev (stanev@ece.upatras.gr)
import os
import pickle
import numpy as np
from tqdm import tqdm
from opensim_utils import calculate_muscle_data, getMuscleIndices, \
    getCoordinateIndices
from util import null_space, construct_muscle_space_inequality, \
    convex_bounded_vertex_enumeration,  readMotionFile, \
    index_containing_substring, write_as_sto_file

###############################################################################
# parameters

# when computed once results are stored into files so that they can be loaded
# with (pickle)
compute = False
subject_dir = os.getcwd() + '/../data/gait1018/'
model_file = subject_dir + 'subject01_scaled.osim'
ik_file = os.getcwd() + '/results/subject01_walk1_InverseKinematics.mot'
id_file = os.getcwd() + '/results/subject01_walk1_InverseDynamics.sto'
so_file = os.getcwd() + '/results/subject01_walk1_StaticOptimization_force.sto'
feasible_set_dir = os.getcwd() + '/results/feasible_force_set/'
results_dir = os.getcwd() + '/results/'

# read opensim files
if not (os.path.isfile(model_file) and
        os.path.isfile(ik_file) and
        os.path.isfile(id_file) and
        os.path.isfile(so_file)):
    raise RuntimeError('required files do not exist')

if not os.path.isdir(feasible_set_dir):
    raise RuntimeError('required folders do not exist')

###############################################################################

if compute:
    ik_header, ik_labels, ik_data = readMotionFile(ik_file)
    ik_data = np.array(ik_data)
    id_header, id_labels, id_data = readMotionFile(id_file)
    id_data = np.array(id_data)
    moment_arm, max_force = calculate_muscle_data(model_file, ik_file)
    model_coordinate_indices, id_coordinate_indices = getCoordinateIndices(
        model_file, id_labels, '^pelvis_.*|^lumbar_.*')
    muscle_indices = getMuscleIndices(model_file, 'do_not_exclude_any_muscle')
    print('Active id coordinates: ', id_coordinate_indices)
    print('Active model coordinates: ', model_coordinate_indices)
    print('Actuve muscles: ', muscle_indices)

    time = ik_data[:, 0]
    entries = time.shape[0]

    # id may have different sampling than ik data
    id_data = np.vstack([time,
                         [np.interp(time, id_data[:, 0], id_data[:, i])
                          for i in range(1, id_data.shape[1])]]).transpose()
    assert(id_data.shape[0] == ik_data.shape[0])

    # collect quantities for computing the feasible muscle forces
    NR = []
    Z = []
    b = []
    fm_par = []
    print('Collecting data ...')
    for t in tqdm(range(entries)):
        # get tau, R, Fmax
        tau = id_data[t, id_coordinate_indices]
        RT = moment_arm[t, model_coordinate_indices, :]
        RT = RT[:, muscle_indices]
        RBarT = np.linalg.pinv(RT)
        fmax = max_force[t, muscle_indices]

        NR_temp = null_space(RT)
        fm_par_temp = - RBarT.dot(tau)
        Z_temp, b_temp = construct_muscle_space_inequality(NR_temp,
                                                           fm_par_temp,
                                                           fmax)

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
    with open(results_dir + 'f_set.dat', 'wb') as fo:
        pickle.dump(f_set, fo)

else:  # store the feasible set into multiple .sto files
    # load data
    with open(results_dir + 'f_set.dat', 'rb') as f_s:
        f_set = pickle.load(f_s)

    so_header, so_labels, so_data = readMotionFile(so_file)
    ik_header, ik_labels, ik_data = readMotionFile(ik_file)
    ik_data = np.array(ik_data)
    time = ik_data[:, 0]
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
        for i in range(0, len(f_set)):
            data_temp.append(f_set[i][j % len(f_set[i])])

        write_as_sto_file(feasible_set_dir + str(j) + '.sto', labels,
                          time, np.array(data_temp))
