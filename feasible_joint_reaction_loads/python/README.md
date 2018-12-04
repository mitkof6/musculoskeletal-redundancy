Description
---

The importance of evaluating the feasible muscle forces is demonstrated in the
context of joint reaction analysis. An accurate estimation of the muscle forces
is essential for the assessment of joint reaction loads. Consequently, the null
space muscle forces can significantly alter the reaction forces without
affecting the movement. Please open the following notebook to run the algorithm:

[Feasible Joint Reaction Loads](feasible_joint_reaction_loads.ipynb)

Scripts for performing the extended joint reaction analysis are also
provided. The script execution must follow the following order:

1. *perform_opensim_analysis.py* - run inverse kinematics, static optimization
and joint reaction analysis on the desired dataset.

2. *calculate_feasible_muscle_forces.py* - using the results from the previous
step calculates the feasible muscle forces that satisfies the action and the
physiological muscle constraints.

3. *perform_joint_reaction_batch.py* - performs joint reaction analysis for each
distinct muscle force solution obtained in the previous step.

4. *analyze_joint_reaction_forces.py* - post process the joint reaction loads
and obtain the min/max joint bounds.


Dependencies
---

The scripts are compatible with python 2. Dependencies can be installed through
python package manager (pip). The following libraries were used in the project:

- OpenSim: python wrappings [Tested Version](https://github.com/mitkof6/opensim-core/tree/stable_2)
- sympy: `pip install sympy`
- numpy: `pip install numpy`
- matplotlib: `pip install matplotlib`
- pandas (`pip install pandas`)
- seaborn (`pip install seaborn`)
- tqdm: `pip install tqdm` for progress bar
- pycddlib (`pip install pycddlib`) for finding the feasible muscle force set
- cython (for pycddlib)
- lrs (command line tool build ../lrslib-062 or install from package manager)
