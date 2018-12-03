# This script automates the execution of Inverse Kinematics, Static Optimization
# and Joint Reaction Analysis. Instead of using the OpenSim GUI one can provide
# the required files and generate the files needed for the analysis.
#
# @author Dimitar Stanev (jimstanev@gmail.com)
#
import os
from opensim_utils import perform_ik, perform_so, perform_jra, plot_sto

###############################################################################
# parameters

subject_dir = os.getcwd() + '/../dataset/Gait10dof18musc/'
model_file = subject_dir + 'subject01.osim'
trc_file = subject_dir + 'subject01_walk.trc'
grf_file = subject_dir + 'subject01_walk_grf.mot'
grf_xml_file = subject_dir + 'subject01_walk_grf.xml'
reserve_actuators_file = subject_dir + 'reserve_actuators.xml'
results_dir = subject_dir + 'notebook_results/'

if not (os.path.isfile(model_file) and
        os.path.isfile(trc_file) and
        os.path.isfile(grf_file) and
        os.path.isfile(grf_xml_file) and
        os.path.isfile(reserve_actuators_file)):
    raise RuntimeError('required files do not exist')

if not os.path.isdir(results_dir):
    raise RuntimeError('required folders do not exist')

###############################################################################
# main

# perform OpenSim inverse kinematics
ik_file = perform_ik(model_file, trc_file, results_dir)
# plot_sto(ik_file, 4, save=True)

# perform OpenSim static optimization
(so_force_file, so_activation_file) = perform_so(model_file, ik_file, grf_file,
                                                 grf_xml_file,
                                                 reserve_actuators_file,
                                                 results_dir)
# plot_sto(so_force_file, 4, save=True)

# perform OpenSim joint reaction analysis
jra_file = perform_jra(model_file, ik_file, grf_file, grf_xml_file,
                       reserve_actuators_file, so_force_file, results_dir)
