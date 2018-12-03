FROM ubuntu:latest

# required for tzdata
ENV DEBIAN_FRONTEND=noninteractive

# install ubuntu packages
RUN apt-get update && apt-get -y upgrade && apt-get -y \
install imagemagick python python-pip cython libgmp-dev \
lrslib git python-tk liblapack-dev freeglut3-dev libxi-dev \
libxmu-dev doxygen swig python-dev g++ openjdk-11-jdk \
cmake-curses-gui tzdata nano

# setup timezone
RUN ln -fs /usr/share/zoneinfo/Europe/Athens /etc/localtime
RUN dpkg-reconfigure --frontend noninteractive tzdata

# setup python dependencies
RUN pip install scipy numpy sympy pandas seaborn matplotlib \
pycddlib tqdm pydy jupyter

# install and setup OpenSim
ADD opensim-core-stable2.tar.xz /root/
ENV SIMBODY_HOME=/root/opensim-core-stable2/dep/simbody
ENV Simbody_DIR=$SIMBODY_HOME/lib/cmake/simbody
ENV PATH=$SIMBODY_HOME/libexec/simbody:$PATH
ENV LD_LIBRARY_PATH=$SIMBODY_HOME/lib:$LD_LIBRARY_PATH
ENV OPENSIM_HOME=/root/opensim-core-stable2
ENV OpenSim_DIR=$OPENSIM_HOME/lib/cmake/OpenSim
ENV LD_LIBRARY_PATH=$OPENSIM_HOME/lib:$LD_LIBRARY_PATH
RUN cd /root/opensim-core-stable2/lib/python2.7/site-packages \
&& python setup.py install

# copy demos to default directory
ADD arm_model.tar.xz /root/
ADD feasible_joint_reaction_loads.tar.xz /root/

# make port 80 available to the outside
EXPOSE 80

# setup jupyter (with password 'stanev')
RUN cd && mkdir .jupyter
ADD jupyter_notebook_config.json /root/.jupyter/jupyter_notebook_config.json
CMD cd && jupyter notebook --ip=0.0.0.0 --port=80 --allow-root

