###############
#  CONTENTS   #
###############

The package "paradiseo_tutorials" contains pre-compiled files for different operating systems and a "not_compiled" package:
 - windows
 - linux 32bits
 - linux 64bits
 - not_compiled

Each pre_compiled sub-directory contains an executable file, a parameter file and a c++ source file (provided as an informational resource only) for each lesson:
 - EO_lesson1 (contains "tsp_EA(.exe)", "tsp_EA.cpp", param)
 - MO_lesson1 (contains "hill_climbing(.exe)", "hill_climbing.cpp", param)
 - MO_lesson2 (contains "tabu_search(.exe)", "tabu_search.cpp", param)
 - MO_lesson3 (contains "simulated_annealing(.exe)", "simulated_annealing.cpp", param)
 - MO_lesson4 (contains "iterated_local_search(.exe)", "iterated_local_search.cpp", param)
 - hybrid_lesson (contains "hybrid_ga(.exe)", "hybrid_ga.cpp", param)
 - MOEO_lesson (contains "FlowShopEA(.exe)", "FlowShopEA.cpp", param)

For advance users:
 - The package "not_compiled" allows you to compile tutorials in your own machine ONLY IF PARADISEO HAS BEEN PREVIOUSLY INSTALLED on it.



###############
#     USE     #
###############

Copy the package corresponding to your operating system on your computer.

Then, you can execute all lessons with a command line interpreter.
Windows users, do not click on the executable file "*.exe" whereas you won't be able to see the results.

For instance, to run the hill_climbing, launch a command line interpreter, go to the "MO_lesson1" directory and type:
(windows system)
   > hill_climbing.exe @param
(Linux system)
   > ./hill_climbing @param



###############
# COMPILATION #
###############

We recommend you to use the pre-compiled packages. But if you still want to compile the "not_compiled" package, please perform the following steps.

Linux users:
************
1. Go to the not_compiled directory
2. Edit the install.cmake file 
   • PARADISEO DIR : replace "TO FILL" by the path where ParadisEO has been installed (for instance, "/home/user/paradiseo-1.1/")
   • SOURCES DIR : replace "TO FILL" by the path where the install.cmake file is located on your computer (for instance, "/home/user/tutorials/not_compiled/")
3. Go to the build directory and run the following command lines:
   > cmake ..
   > make
   > make install

Windows users (Visual Studio 9 2008):
*************************************
1. Go to the not_compiled directory
2. Edit the install.cmake file 
   • PARADISEO DIR : replace "TO FILL" by the path where ParadisEO has been installed WITH DOUBLE BACKSLASHES (for instance, "C:\\[ParadisEO_PATH]")
   • SOURCES DIR : replace "TO FILL" by the path where the install.cmake file is located on your computer (for instance, "C:\\...\\not_compiled")
3. Run The Cmake interface.
   • In the field "Where is the source code:", browse to find path of the "not_compiled" directory.
   • In the field "Where to build the binaries:", browse to find path of the "not_compiled/build" directory.
   • Click on "Configure"
   • Choose "Visual Studio 9 2008"
   • Skip Warnings (click "OK" for all)
   • click on "Configure"
   • Skip Warnings (click "OK" for all)
   • Click on "Ok"
   • Skip Warnings (click "OK" for all)
4. Compilation.
   • Go in "not_compiled/build" directory.
   • Double click on the Visual Studio Solution "TUTORIAUX".
   • Skip Warnings while Visual Studio is launched (click "OK" for all)
   • Choose "Release" in the top of the windows near the green arrow.
   • In the Solutions Explorer (on the left), right click on "Solution'TUTORIAUX'", then click on "Build Solution".
   • right click on "installall", then click on "Build Solution".
Now lessons should be compiled in the build directories.
Executable are in the "Release" directories.
NOTE: Relative path in "param" files must be changed in added "../" to --instancePath.
