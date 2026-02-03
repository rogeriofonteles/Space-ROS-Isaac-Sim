# hadolint ignore=DL3007
FROM osrf/space-ros:humble-2024.10.0 AS base

# Define arguments used in the metadata definition
ARG VCS_REF
ARG VERSION="preview"

# Specify the docker image metadata
LABEL org.label-schema.schema-version="1.0"
LABEL org.label-schema.name="Space-ROS-NVIDIA-Isaac-Sim Curiosity's Sulfur Stone Discovery"
LABEL org.label-schema.description="Repository provides a Docker image for NASA's Curiosity rover demo"
LABEL org.label-schema.vendor="Open Robotics"
LABEL org.label-schema.version=${VERSION}
LABEL org.label-schema.url="https://github.com/space-ros"
LABEL org.label-schema.vcs-url="https://github.com/space-ros/docker"
LABEL org.label-schema.vcs-ref=${VCS_REF}

FROM base

ENV DEBIAN_FRONTEND=noninteractive

USER root

COPY --link apt-requirements.txt /apt-requirements.txt
# hadolint ignore=SC2046
RUN apt-get update && \
    apt-get install --no-install-recommends -y $(cat /apt-requirements.txt) && \
    rm -rf /var/lib/apt/lists/*
RUN rm /apt-requirements.txt

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

RUN apt-add-repository ppa:fish-shell/release-3 && \
      apt-get update && \
      apt-get install -y fish

COPY --link ros-requirements.txt /ros-requirements.txt
# hadolint ignore=SC2046
RUN apt-get update && \
    apt-get install --no-install-recommends -y $(cat /ros-requirements.txt) && \
    rm -rf /var/lib/apt/lists/* && \
    rm /ros-requirements.txt

RUN locale-gen en_US en_US.UTF-8 &&  update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

ENV LANG=en_US.UTF-8

# Install Isaac Sim
COPY --link nvidia-pip-requirements.txt /requirements.txt
RUN pip install --timeout=600 --no-cache-dir -r /requirements.txt --extra-index-url https://pypi.nvidia.com && \
    rm /requirements.txt

# Remove old botocore/boto3/s3transfer
RUN rm -rf /usr/local/lib/python3.10/dist-packages/isaacsim/extscache/omni.kit.pip_archive/pip_prebundle/botocore && \
    rm -rf /usr/local/lib/python3.10/dist-packages/isaacsim/extscache/omni.kit.pip_archive/pip_prebundle/boto3 && \
    rm -rf /usr/local/lib/python3.10/dist-packages/isaacsim/extscache/omni.kit.pip_archive/pip_prebundle/s3transfer

RUN adduser $USERNAME sudo && echo "$USERNAME ALL=NOPASSWD: ALL" >> /etc/sudoers.d/$USERNAME

RUN mkdir -p /usr/local/lib/python3.10/dist-packages/isaacsim/extscache/omni.gpu_foundation/cache && \
    mkdir -p /usr/local/lib/python3.10/dist-packages/omni/logs && \
    mkdir -p /usr/local/lib/python3.10/dist-packages/omni/cache/Kit/106.0/fcefe91f/PhysXLocalMeshCache && \
    mkdir -p /usr/local/lib/python3.10/dist-packages/omni/data/documents/Kit/apps/Isaac-Sim/scripts/new_stage && \
    mkdir -p /usr/local/lib/python3.10/dist-packages/omni/data/documents/Kit/shared/screenshots && \
    chown -R $USERNAME:$USERNAME /usr/local/lib/python3.10/dist-packages/isaacsim/extscache/omni.gpu_foundation/cache && \
    chown -R $USERNAME:$USERNAME /usr/local/lib/python3.10/dist-packages/omni/logs && \
    chown -R $USERNAME:$USERNAME /usr/local/lib/python3.10/dist-packages/omni/cache && \
    chown -R $USERNAME:$USERNAME /usr/local/lib/python3.10/dist-packages/omni/data/documents && \
    chmod -R 777 /usr/local/lib/python3.10/dist-packages/isaacsim/* && \
    chmod -R 777 /usr/local/lib/python3.10/dist-packages/omni/logs && \
    chmod -R 777 /usr/local/lib/python3.10/dist-packages/omni/cache && \
    chmod -R 777 /usr/local/lib/python3.10/dist-packages/omni/data/documents

USER ${USERNAME}

# Copy the dependency.repos
COPY --chown=${USERNAME}:${USERNAME} dependency.repos /tmp/dependency.repos

RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    mkdir -p /home/$USERNAME/isaac_ws/src && \
    cd /home/$USERNAME/isaac_ws && \
    vcs import src < /tmp/dependency.repos --recursive && \
    colcon build && \
    rm /tmp/dependency.repos && \
    cd /home/$USERNAME && git clone https://github.com/pojenwang/curiosity_sim.git"

# Set environment in bashrc for future sessions
RUN echo "source /opt/ros/humble/setup.bash" >> /home/"$USERNAME"/.bashrc && \
    echo "source /home/$USERNAME/isaac_ws/install/setup.bash" >> /home/"$USERNAME"/.bashrc && \
    echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> /home/"$USERNAME"/.bashrc

# Install Fisher package manager for fish
RUN fish -c "curl -sL https://raw.githubusercontent.com/jorgebucaran/fisher/main/functions/fisher.fish | source && fisher install jorgebucaran/fisher"
# Install bass to use bash commands in fish
RUN fish -c "fisher install edc/bass"
# Install ros fish auto complete
RUN mkdir -p ~/.config/fish/conf.d ~/.config/fish/completions ~/.config/fish/functions && \
      curl -sL https://github.com/kpbaks/ros2.fish/archive/refs/heads/main.tar.gz | tar xz -C /tmp && \
      cp /tmp/ros2.fish-main/conf.d/* ~/.config/fish/conf.d/ && \
      cp /tmp/ros2.fish-main/completions/* ~/.config/fish/completions/ && \
      cp /tmp/ros2.fish-main/functions/* ~/.config/fish/functions/ && \
      sed -i '1s/status is-interactive; or return 1/if not status is-interactive\n    return 0\nend/' ~/.config/fish/conf.d/ros2.fish && \
      rm -rf /tmp/ros2.fish-main

COPY --link entrypoint.sh /.entrypoint.sh
ENTRYPOINT ["/.entrypoint.sh"]
