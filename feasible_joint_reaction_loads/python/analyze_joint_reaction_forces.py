# This script processes the joint reaction loads and computes the minimum and
# maximum bounds for a given joint of interest. The subject directory, the
# folder containing the joint reaction results form
# perform_joint_reaction_batch.py, the joint of interest and the mass of the
# subject must be provided.
#
# @author Dimitar Stanev (jimstanev@gmail.com)
#
import os
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
from tqdm import tqdm
from util import readMotionFile, index_containing_substring

###############################################################################
# parameters

subject_dir = os.getcwd() + '/../dataset/Gait10dof18musc/'
os_jra_file = subject_dir + 'notebook_results/subject01_walk_JointReaction_ReactionLoads.sto'
jra_results_dir = subject_dir + 'notebook_results/joint_reaction_analyses/'
figures_dir = subject_dir + 'notebook_results/fig/'

collect = True
use_abs = False
joint = 'hip_l'
joints = 3
mass = 72.6  # kg
g = 9.8  # m/s^2
body_weight = mass * g

if not os.path.isfile(os_jra_file):
    raise RuntimeError('required files do not exist')

if not (os.path.isdir(jra_results_dir) and
        os.path.isdir(figures_dir)):
    raise RuntimeError('required folders do not exist')

###############################################################################
# main

# OpenSim's JRA results
os_header, os_labels, os_data = readMotionFile(os_jra_file)
os_data = np.array(os_data)
joint_index = index_containing_substring(os_labels, joint)
assert(joint_index != -1)

# get all files in the directory
jra_files = os.listdir(jra_results_dir)
# remove files that are not joint reactions
jra_files = [e for e in jra_files if 'ReactionLoads' in e]

if collect:
    # collect simulation data
    print('Processing joint reaction analyses ...')
    # allocate the necessary space
    solutions_to_keep = len(jra_files)
    simulationData = np.empty([solutions_to_keep,
                               os_data.shape[0],
                               os_data.shape[1]],
                              dtype=float)
    # collect data
    for i, f in enumerate(tqdm(jra_files)):
        if i == solutions_to_keep:
            break

        header, labels, data = readMotionFile(jra_results_dir + f)
        simulationData[i, :, :] = np.array(data)


if use_abs:
    simulationData = np.abs(simulationData)
    os_data = np.abs(os_data)

# def _plot_range_band(central_data=None, ci=None, data=None, *args, **kwargs):
#     upper = data.max(axis=0)
#     lower = data.min(axis=0)
#     #import pdb; pdb.set_trace()
#     ci = np.asarray((lower, upper))
#     kwargs.update({"central_data": central_data, "ci": ci, "data": data})
#     sns.timeseries._plot_ci_band(*args, **kwargs)

# sns.timeseries._plot_range_band = _plot_range_band


heel_strike_right = [0.65, 1.85]
toe_off_right = [0.15, 1.4]
heel_strike_left = [0.0, 1.25]
toe_off_left = [0.8, 2]
# heel_strike_right = [0.65]
# toe_off_right = [1.4]
# heel_strike_left = [1.25]
# toe_off_left = [0.8]
if '_l' in joint:
    heel_strike = heel_strike_left
    toe_off = toe_off_left
else:
    heel_strike = heel_strike_right
    toe_off = toe_off_right

# plot data min/max reactions vs OpenSim JRA
fig, ax = plt.subplots(nrows=1, ncols=joints, figsize=(15, 5))
for i in range(0, joints):
    # sns.tsplot(time=os_data[1:, 0],
    #            data=simulationData[1:, 1:, joint_index[i]] / body_weight,
    #            ci=[0, 100], err_style='range_band', color='b', ax=ax[i])
    # plot feasible reaction loads
    min_reaction = np.min(
        simulationData[1:, 1:, joint_index[i]] / body_weight, axis=0)
    max_reaction = np.max(
        simulationData[1:, 1:, joint_index[i]] / body_weight, axis=0)
    ax[i].fill_between(os_data[1:, 0], min_reaction, max_reaction, color='b',
                       alpha=0.2, label='Feasible Reactions')
    # plot OpenSim reaction loads
    ax[i].plot(os_data[1:, 0], os_data[1:, joint_index[i]] / body_weight,
               '-.r', label='OpenSim JRA')
    # annotate the heel strike and toe off regions
    min_min = np.min(min_reaction)
    max_max = np.max(max_reaction)
    ax[i].vlines(x=heel_strike, ymin=min_min, ymax=max_max,
                 color='c', linestyle='--', label='HS')
    ax[i].vlines(x=toe_off, ymin=min_min, ymax=max_max,
                 color='m', linestyle=':', label='TO')
    # figure settings
    ax[i].set_title(os_labels[joint_index[i]])
    ax[i].set_xlabel('time (s)')
    ax[i].set_ylabel('reaction / body weight')
    if i == joints - 1:
        ax[i].legend()


fig.tight_layout()
fig.savefig(figures_dir + joint + '.pdf',
            format='pdf', dpi=300)
fig.savefig(figures_dir + joint + '.png',
            format='png', dpi=300)
fig.show()
