# \brief A variety of useful OpenSim utilities.
#
# @author Dimitar Stanev (jimstanev@gmail.com)
#
import os
import opensim
from tqdm import tqdm
import numpy as np
from util import plot_sto


def calculate_muscle_data(model_file, ik_file):
    """This function calculates the moment arm and maximum muscle force provided an
    OpenSim model and a motion from inverse kinematics.

    Parameters
    ----------
    model_file: string
        OpenSim .osim file
    ik_file: string
        .mot from inverse kinematics

    Returns
    -------
    tuple:
        (moment arm[time x coordinates x muscles], max_force[time x muscles])
    """
    model = opensim.Model(model_file)
    state = model.initSystem()

    # model coordinates dictionary
    model_coordinates = {}
    for i in range(0, model.getNumCoordinates()):
        model_coordinates[model.getCoordinateSet().get(i).getName()] = i

    # model muscles dictionary
    model_muscles = {}
    for i in range(0, model.getNumControls()):
        model_muscles[model.getMuscles().get(i).getName()] = i

    # process the motion
    motion = opensim.Storage(ik_file)
    labels = motion.getColumnLabels()
    isInDeg = motion.isInDegrees()

    # calculate moment arm and max force
    max_force = np.empty([motion.getSize(), len(model_muscles)], float)
    moment_arm = np.empty([motion.getSize(),
                           len(model_coordinates),
                           len(model_muscles)],
                          float)
    for t in tqdm(range(0, motion.getSize())):
        state_vector = motion.getStateVector(t)
        state_data = state_vector.getData()

        # update model pose
        for j in range(0, state_data.getSize()):
            coordinate = model_coordinates[labels.get(j + 1)]  # time is 0
            if isInDeg:
                value = np.deg2rad(state_data.get(j))
            else:
                value = state_data.get(j)

            model.updCoordinateSet().get(coordinate).setValue(state, value)

        model.realizePosition(state)

        # calculate muscle moment arm
        for coordinate_index in model_coordinates.values():
            for muscle_index in model_muscles.values():
                coordinate = model.getCoordinateSet().get(coordinate_index)
                muscle = model.getMuscles().get(muscle_index)
                ma = muscle.computeMomentArm(state, coordinate)
                moment_arm[t, coordinate_index, muscle_index] = ma

        # calculate max force (TODO use force-length/velocity relations)
        for muscle_index in model_muscles.values():
            muscle = model.getMuscles().get(muscle_index)
            max_force[t, muscle_index] = muscle.getMaxIsometricForce()

    return (moment_arm, max_force)


def construct_ik_task_set(model, marker_data, task_set):
    """Construct OpenSim Inverse Kinematics task set.

    In older versions of OpenSim (e.g. 3.3) IK will not execute when there are
    virtual markers that do not exist in the marker data.

    """
    virtual_markers = model.getMarkerSet()
    marker_names = marker_data.getMarkerNames()
    for i in range(0, marker_names.getSize()):
        marker_name = marker_names.get(i)
        exists = False
        for j in range(0, virtual_markers.getSize()):
            if virtual_markers.get(j).getName() == marker_name:
                task = opensim.IKMarkerTask()
                task.setName(marker_name)
                task.setApply(True)
                task.setWeight(1)
                task_set.adoptAndAppend(task)
                exists = True
                break

        if not exists:
            task = opensim.IKMarkerTask()
            task.setName(marker_name)
            task.setApply(False)
            task.setWeight(1)
            task_set.adoptAndAppend(task)


def perform_ik(model_file, trc_file, results_dir):
    """Performs Inverse Kinematics using OpenSim.

    Parameters
    ----------
    model_file: str
        OpenSim model (.osim)
    trc_file: str
        the experimentally measured marker trajectories (.trc)
    results_dir: str
        directory to store the results

    """
    model = opensim.Model(model_file)
    model.initSystem()
    marker_data = opensim.MarkerData(trc_file)
    name = os.path.basename(trc_file)[:-4]
    ik_tool = opensim.InverseKinematicsTool()
    task_set = ik_tool.getIKTaskSet()
    construct_ik_task_set(model, marker_data, task_set)
    ik_tool.setName(name)
    ik_tool.setModel(model)
    ik_tool.setStartTime(marker_data.getStartFrameTime())
    ik_tool.setEndTime(marker_data.getLastFrameTime())
    ik_tool.setMarkerDataFileName(trc_file)
    ik_tool.setResultsDir(results_dir)
    ik_file = results_dir + name + '_ik.mot'
    ik_tool.setOutputMotionFileName(ik_file)
    ik_tool.run()
    return ik_file


def visualize_ik_results(ik_file, ik_errors=None):
    """A utility for visualizing the Inverse Kinematics results.

    Parameters
    ----------
    ik_file: str
         coordinate results from IK (.mot)
    ik_errors: str, optional
         marker errors (.sto)
    """
    plot_sto(ik_file, 4)
    if ik_errors is not None:
        plot_sto(ik_errors, 3)


def perform_so(model_file, ik_file, grf_file, grf_xml, reserve_actuators,
               results_dir):
    """Performs Static Optimization using OpenSim.

    Parameters
    ----------
    model_file: str
        OpenSim model (.osim)
    ik_file: str
        kinematics calculated from Inverse Kinematics
    grf_file: str
        the ground reaction forces
    grf_xml: str
        xml description containing how to apply the GRF forces
    reserve_actuators: str
        path to the reserve actuator .xml file
    results_dir: str
        directory to store the results
    """
    # model
    model = opensim.Model(model_file)

    # prepare external forces xml file
    name = os.path.basename(grf_file)[:-8]
    external_loads = opensim.ExternalLoads(model, grf_xml)
    external_loads.setExternalLoadsModelKinematicsFileName(ik_file)
    external_loads.setDataFileName(grf_file)
    external_loads.setLowpassCutoffFrequencyForLoadKinematics(6)
    external_loads.printToXML(results_dir + name + '.xml')

    # add reserve actuators
    force_set = opensim.ForceSet(model, reserve_actuators)
    force_set.setMemoryOwner(False)  # model will be the owner
    for i in range(0, force_set.getSize()):
        model.updForceSet().append(force_set.get(i))

    # construct static optimization
    motion = opensim.Storage(ik_file)
    static_optimization = opensim.StaticOptimization()
    static_optimization.setStartTime(motion.getFirstTime())
    static_optimization.setEndTime(motion.getLastTime())
    static_optimization.setUseModelForceSet(True)
    static_optimization.setUseMusclePhysiology(True)
    static_optimization.setActivationExponent(2)
    static_optimization.setConvergenceCriterion(0.0001)
    static_optimization.setMaxIterations(100)
    model.addAnalysis(static_optimization)

    # analysis
    analysis = opensim.AnalyzeTool(model)
    analysis.setName(name)
    analysis.setModel(model)
    analysis.setInitialTime(motion.getFirstTime())
    analysis.setFinalTime(motion.getLastTime())
    analysis.setLowpassCutoffFrequency(6)
    analysis.setCoordinatesFileName(ik_file)
    analysis.setExternalLoadsFileName(results_dir + name + '.xml')
    analysis.setLoadModelAndInput(True)
    analysis.setResultsDir(results_dir)
    analysis.run()
    so_force_file = results_dir + name + '_StaticOptimization_force.sto'
    so_activations_file = results_dir + name + \
                          '_StaticOptimization_activation.sto'
    return (so_force_file, so_activations_file)


def visualize_so_results(activations_file, forces_file=None):
    """A utility for visualizing the Static Optimization results.

    Parameters
    ----------
    activations_file: str
         activations results from SO (.sto)
    forces_file: str, optional
         forces (.sto)
    """
    plot_sto(activations_file, 8, '_r')
    if forces_file is not None:
        plot_sto(forces_file, 8, '_r')


def perform_jra(model_file, ik_file, grf_file, grf_xml, reserve_actuators,
                muscle_forces_file, results_dir, prefix=''):
    """Performs Static Optimization using OpenSim.

    Parameters
    ----------
    model_file: str
        OpenSim model (.osim)
    ik_file: str
        kinematics calculated from Inverse Kinematics
    grf_file: str
        the ground reaction forces
    grf_xml: str
        xml description containing how to apply the GRF forces
    reserve_actuators: str
        path to the reserve actuator .xml file
    muscle_forces_file: str
        path to the file containing the muscle forces from SO
    results_dir: str
        directory to store the results
    prefix: str
        prefix of the resultant joint reaction loads
    """
    # model
    model = opensim.Model(model_file)

    # prepare external forces xml file
    name = os.path.basename(grf_file)[:-8]
    external_loads = opensim.ExternalLoads(model, grf_xml)
    external_loads.setExternalLoadsModelKinematicsFileName(ik_file)
    external_loads.setDataFileName(grf_file)
    external_loads.setLowpassCutoffFrequencyForLoadKinematics(6)
    external_loads.printToXML(results_dir + name + '.xml')

    # TODO this may not be needed
    # add reserve actuators (must not be appended when performing JRA)
    # force_set = opensim.ForceSet(model, reserve_actuators)
    # force_set.setMemoryOwner(False)  # model will be the owner
    # for i in range(0, force_set.getSize()):
    #     model.updForceSet().append(force_set.get(i))
    #     # model.addForce(force_set.get(i))

    # construct joint reaction analysis
    motion = opensim.Storage(ik_file)
    joint_reaction = opensim.JointReaction(model)
    joint_reaction.setName('JointReaction')
    joint_reaction.setStartTime(motion.getFirstTime())
    joint_reaction.setEndTime(motion.getLastTime())
    joint_reaction.setForcesFileName(muscle_forces_file)
    model.addAnalysis(joint_reaction)
    model.initSystem()

    # analysis
    analysis = opensim.AnalyzeTool(model)
    analysis.setName(prefix + name)
    analysis.setModel(model)
    analysis.setModelFilename(model_file)
    analysis.setInitialTime(motion.getFirstTime())
    analysis.setFinalTime(motion.getLastTime())
    analysis.setLowpassCutoffFrequency(6)
    analysis.setCoordinatesFileName(ik_file)
    analysis.setExternalLoadsFileName(results_dir + name + '.xml')
    analysis.setLoadModelAndInput(True)
    analysis.setResultsDir(results_dir)
    analysis.run()
    jra_file = results_dir + name + '_JointReaction_ReactionLoads.sto'
    return jra_file
