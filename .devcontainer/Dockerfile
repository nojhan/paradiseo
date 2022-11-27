FROM gcc:10

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \ 
    apt-get install -y --no-install-recommends sudo curl vim git htop build-essential \
    gdb cppcheck clang-tidy ccache gcovr pkg-config doxygen \
    cmake make libeigen3-dev libopenmpi-dev doxygen graphviz libgnuplot-iostream-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && useradd -ms /bin/bash paradiseo \
    && chmod 660 /etc/sudoers && echo "paradiseo ALL=NOPASSWD: ALL" >> /etc/sudoers && chmod 400 /etc/sudoers

WORKDIR /workspaces

RUN chown -R paradiseo:paradiseo /workspaces