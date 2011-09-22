This package contains the source for ParadisEO-GPU problems.

# Step 1 - Configuration
------------------------
Edit the "install.cmake" file by entering the FULL PATH of :

-"ParadisEO_PATH" where Paradiseo directory has been installed in your host.
-"ParadisEO-GPU_PATH" where ParadisEO-GPU package has been decompressed in your host.
-"CUDA_PATH" where CUDA has been installed in your host.
-"NVIDIA_PATH" where NVIDIA has been installed in your host.


# Step 2 - Build process
------------------------
ParadisEO is assumed to be compiled. To download ParadisEO, please visit http://paradiseo.gforge.inria.fr/.
Go to the ParadisEO-GPU/build/ directory and lunch cmake:
(Unix)       > cmake .. -DENABLE_CMAKE_TESTING=TRUE -DCMAKE_BUILD_TYPE=Debug


# Step 3 - Compilation
----------------------
In the ParadisEO-GPU/build/ directory:
(Unix)       > make

# Step 4 - Execution
---------------------
A toy example is given to test the components. You can run these tests as following.
To define problem-related components for your own problem, please refer to the tutorials available on the website : http://paradiseo.gforge.inria.fr/.
In the ParadisEO-GPU/build/ directory:
(Unix)       > ctest -D ExperimentalStart -D ExperimentalBuild -D ExperimentalTest -D ExperimentalSubmit

In the directory "tutorial", there is an example of One Max problem which illustrate how to use this package.

# Documentation
---------------
The API-documentation is available in doc/html/index.html 

