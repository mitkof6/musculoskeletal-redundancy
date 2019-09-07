# This script automates the execution of Inverse Kinematics, Static Optimization
# and Joint Reaction Analysis. Instead of using the OpenSim GUI one can provide
# the required files and generate the files needed for the analysis.
#
# @author Dimitar Stanev (stanev@ece.upatras.gr)
#
import os
from opensim_utils import perform_ik, perform_id, perform_so, perform_jra
from util import plot_sto

###############################################################################
# parameters

subject_dir = os.getcwd() + '/../data/gait1018/'
model_file = subject_dir + 'subject01_scaled.osim'
trc_file = subject_dir + 'subject01_walk1.trc'
grf_file = subject_dir + 'subject01_walk1_grf.mot'
grf_xml_file = subject_dir + 'subject01_walk1_grf.xml'
reserve_actuators_file = subject_dir + 'subject01_actuators.xml'
results_dir = os.getcwd() + '/results/'

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
plot_sto(ik_file, 2, os.path.join(results_dir, ik_file[0:-4] + '.pdf'))

# perform OpenSim inverse dynamics
id_file = perform_id(model_file, ik_file, grf_file, grf_xml_file, results_dir)
plot_sto(id_file, 2, os.path.join(results_dir, id_file[0:-4] + '.pdf'))

# perform OpenSim static optimization
(so_force_file, so_activation_file) = perform_so(model_file, ik_file, grf_file,
                                                 grf_xml_file,
                                                 reserve_actuators_file,
                                                 results_dir)
plot_sto(so_force_file, 4, os.path.join(results_dir,
                                        so_force_file[0:-4] + '.pdf'))
plot_sto(so_activation_file, 3, os.path.join(results_dir,
                                             so_activation_file[0:-4] + '.pdf'))

# perform OpenSim joint reaction analysis
jra_file = perform_jra(model_file, ik_file, grf_file, grf_xml_file,
                       reserve_actuators_file, so_force_file, results_dir, '',
                       ['ALL'], ['child'], ['ground'])
plot_sto(jra_file, 3, os.path.join(results_dir, jra_file[0:-4] + '.pdf'),
         None,
         lambda x: x.replace('_on_', '\n').replace('_in_', '\n'))
