[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2000421.svg)](https://doi.org/10.5281/zenodo.2000421)

Description
---

This project contains the source code related to the following publication:

D. Stanev and Konstantinos Moustakas, Modeling musculoskeletal kinematic and 
dynamic redundancy using null space projection, PLoS ONE, 14(1): e0209171, 
Jan. 2019, DOI: https://doi.org/10.1371/journal.pone.0209171

The coordination of the human musculoskeletal system is deeply influenced by its
redundant nature, in both kinematic and dynamic terms. Noticing a lack of a
relevant, thorough treatment in the literature, we formally address the issue in
order to understand and quantify factors affecting the motor coordination. We
employed well-established techniques from linear algebra and projection
operators to extend the underlying kinematic and dynamic relations by modeling
the redundancy effects in null space. We found that there are three operational
spaces, namely task, joint and muscle space, which are directly associated with
the physiological factors of the system. A method for consistently quantifying
the redundancy on multiple levels in the entire space of feasible solutions is
also presented. We evaluate the proposed muscle space projection on segmental
level reflexes and the computation of the feasible muscle space forces for
arbitrary movement. The former proves to be a convenient representation for
interfacing with segmental level models or implementing controllers for tendon
driven robots, while the latter enables the identification of force variability
and correlations between muscle groups, attributed to the system's
redundancy. Furthermore, the usefulness of the proposed framework is
demonstrated in the context of estimating the bounds of the joint reaction
loads, where we show that misinterpretation of the results is possible if the
null space forces are ignored. This work presents a theoretical analysis of the
redundancy problem, facilitating application in a broad range of fields related
to motor coordination, as it provides the groundwork for null space
characterization. The proposed framework rigorously accounts for the effects of
kinematic and dynamic redundancy, incorporating it directly into the underlying
equations using the notion of null space projection, leading to a complete
description of the system.

Repository Overview
---

- arm_model: simulation of simple arm model
- feasible_joint_reaction_loads: calculation of the feasible reaction loads, by
  accounting for musculoskeletal redundancy effects
- docker: a self contained docker setup file, which installs all dependencies
  related to the developed algorithms


Demos
---

The user can navigate into the corresponding folders and inspect the source
code. The following case studies are provided in the form of interactive Jupyter
notebooks:

- [Arm Model](arm_model/model.ipynb) presents a case study using muscle space
  projection to study the response of segmental level reflexes

- [Muscle Space Projection](arm_model/muscle_space_projection.ipynb)
  demonstrates muscle space projection in the context of segmental level
  (reflex) modeling

- [Feasible Muscle Forces](arm_model/feasible_muscle_forces.ipynb) uses
  task space projection to simulate a simple hand movement, where the feasible
  muscle forces that satisfy this task are calculated and analyzed

- [Feasible Joint Reaction Loads](feasible_joint_reaction_loads/python/feasible_joint_reaction_loads.ipynb)
  demonstrates the utilization of the feasible muscle forces to calculate the
  bounds of the joint reaction loads during walking

The .html files corresponding to the .ipynb notebooks included in the folders
contain the pre-executed results of the demos.


<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img
alt="Creative Commons License" style="border-width:0"
src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is
licensed under a <a rel="license"
href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution
4.0 International License</a>.
