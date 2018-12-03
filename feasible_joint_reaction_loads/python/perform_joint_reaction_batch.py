# This script performs joint reaction analysis for each feasible muscle forces
# computed in calculate_feasible_muscle_forces.py. The user must provide the
# required files to perform the joint reaction analysis, namely the subject
# directory, model, inverse kinematics motion, ground reaction forces, reserve
# actuators (not used) and the directory containing the feasible muscle forces.
#
# @author Dimitar Stanev (jimstanev@gmail.com)
#
import os
from tqdm import tqdm
from opensim_utils import perform_jra

###############################################################################
# parameters

subject_dir = os.getcwd() + '/../dataset/Gait10dof18musc/'
model_file = subject_dir + 'subject01.osim'
grf_file = subject_dir + 'subject01_walk_grf.mot'
grf_xml_file = subject_dir + 'subject01_walk_grf.xml'
reserve_actuators_file = subject_dir + 'reserve_actuators.xml'
ik_file = subject_dir + 'notebook_results/subject01_walk_ik.mot'
feasible_set_dir = subject_dir + 'notebook_results/feasible_force_set/'
jra_results_dir = subject_dir + 'notebook_results/joint_reaction_analyses/'

if not (os.path.isfile(model_file) and
        os.path.isfile(ik_file) and
        os.path.isfile(grf_file) and
        os.path.isfile(grf_xml_file) and
        os.path.isfile(reserve_actuators_file)):
    raise RuntimeError('required files do not exist')

if not (os.path.isdir(feasible_set_dir) and
       os.path.isdir(jra_results_dir)):
    raise RuntimeError('required folders do not exist')

# get all files in the directory
feasible_force_files = os.listdir(feasible_set_dir)
# remove files that are not .sto
feasible_force_files = [e for e in feasible_force_files if '.sto' in e]

###############################################################################
# main

print('Performing joint reaction analysis batch ...')
previous_iteration = 0
for i, force_file in enumerate(tqdm(feasible_force_files)):
    # Due to memory leaks in OpenSim this script must be restarted. The if
    # statement is used so as to continue from the last performed analysis.
    # change the previous_iteration variable
    if i < previous_iteration:
         continue

    if i > previous_iteration + 300:
        print('please terminate python and rerun this script (RAM usage problem)')
        break

    perform_jra(model_file, ik_file, grf_file, grf_xml_file,
                reserve_actuators_file, feasible_set_dir + force_file,
                jra_results_dir, prefix=str(i) + '_')
