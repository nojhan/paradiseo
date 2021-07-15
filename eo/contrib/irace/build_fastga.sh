#!/bin/sh

########################################################################
# This is an example of how to deal with complex builds,
# for instance on clusters with compilers provided as side modules.
########################################################################

# Run this script in a separate dir, e.g.
# mkdir -p code ; cd code ; ../build_fastga.sh

# exit when any command fails
set -e

# We need recent clang and cmake
module load LLVM/clang-llvm-10.0
module load cmake/3.18

# We are going to use a specific compiler, different from the system's one.
# Path toward the compiler:
C="/opt/dev/Compilers/LLVM/10.0.1/bin"
# Path toward the include for the std lib:
I="/opt/dev/Compilers/LLVM/10.0.1/include/c++/v1/"
# Path toward the compiled std lib:
L="/opt/dev/Compilers/LLVM/10.0.1/lib"

# As we use clang, we use its std lib (instead of gcc's "libstdc++")
S="libc++"

# Gather all those into a set of flags:
flags="-I${I} -stdlib=${S} -L${L}"

# Current dir, for further reference.
here=$(pwd)

# Compiler selection
export CC=${C}/clang
export CXX=${C}/clang++

# If the dir already exists
if cd IOHexperimenter ; then
    # Just update the code
    git pull
else
    # Clone the repo
    git clone --branch feat+EAF --single-branch --recurse-submodules https://github.com/jdreo/IOHexperimenter.git
    cd IOHexperimenter
fi
# Clean build from scratch
rm -rf release
mkdir -p release
cd release
cmake -DCMAKE_CXX_FLAGS="${flags}" -D CMAKE_BUILD_TYPE=Release -D BUILD_TESTS=OFF -D BUILD_EXAMPLE=OFF ..
make -j
# Get back to the root dir
cd ${here}


if cd paradiseo ; then
    git pull
else
    git clone --branch feat+num_foundry --single-branch --recurse-submodules https://github.com/jdreo/paradiseo.git
    cd paradiseo
    touch LICENSE
fi
rm -rf release
mkdir -p release
cd release
cmake -DCMAKE_CXX_FLAGS="${flags}" -D CMAKE_BUILD_TYPE=Release ..
make -j
cd ${here}


cd paradiseo/eo/contrib/irace
rm -rf release
mkdir -p release
cd release
cmake -DCMAKE_CXX_FLAGS="${flags}" -D CMAKE_BUILD_TYPE=Release -D IOH_ROOT=${here}/IOHexperimenter/ -D PARADISEO_ROOT=${here}/paradiseo/ -D PARADISEO_BUILD=${here}/paradiseo/release/ ..
make -j
cd ${here}

