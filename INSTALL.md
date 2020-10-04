
Summary
=======

As Paradiseo is a development framework, you do not really need to install it on all your systems. Just put it somewhere on your development computer, compile it from here and indicate where to find it to your favorite build system.


Build
-----

Paradiseo is mainly developed for Linux, on which it is straightforward to install a C++ build chain. For example, on Ubuntu 18.04:
```bash
sudo apt install g++-8 cmake make libeigen3-dev libopenmpi-dev doxygen graphviz libgnuplot-iostream-dev
```

Paradiseo use the CMake build system, so building it should be as simple as:
```bash
mkdir build ; cd build ; cmake -DEDO=ON .. && make -j
```

Develop
-------

Download the quick start project template, edit the `CMakeLists.txt` file to indicate where to find Paradiseo and start developing your own solver.

If you don't know CMake or a modern build system, you should still be able to build a stand-alone code from a `paradiseo/build` directory with something like:
```bash
 c++ ../solver.cpp -I../eo/src -I../edo/src -DWITH_EIGEN=1 -I/usr/include/eigen3 -std=c++17 -L./lib/ -leo -leoutils -les -lga -o solver
```


Install
-------

If you want to install ParadisEO system-wide anyway:
```bash
cmake -D CMAKE_BUILD_TYPE=Release .. && sudo make install
```


More details
============

As a templated framework, most of the ParadisEO code rely within headers and is thus compiled
by you when you build your own solver.

However, in order to save some compilation time,
the EO and EDO modules are compiled within static libraries by the default build system.

If you believe you have a working build chain and want to test if it works with ParadisEO,
you can try to build the tests and the examples.
Note that if some of them failed (but not all), you may still be able to build your own solver,
as you will most probably not use all ParadisEO features anyway.


Windows
-------

Last time we checked, ParadisEO could only be built with MinGW.
Feel free to test with another compiler and to send us your report.

As of today, we cannot guarantee that it will be easy to
install ParadisEO under Windows if you're a beginner.
There is still some (possibly outdated) help about oldest version on the [Website](http://paradiseo.gforge.inria.fr/).

If you know how to install a working compiler and the dependencies,
you may follow the same steps than the Linux process below.

If you are a beginner, we strongly suggest you install a Linux distribution
(either as an OS, as a virtual machine or using the Windows 10 compatibility layer).


Linux
-----

### Dependencies

In order to build the latest version of Paradiseo, you will need a C++ compiler supporting C++17.
So far, GCC and CLANG gave good results under Linux. You will also need the CMake and make build tools.

Some features are only available if some dependencies are installed:
- Most of the EDO module depends on either uBlas or Eigen3. The recommended package is Eigen3, which enables the adaptive algorithms.
- Doxygen is needed to build the API documentation, and you should also install graphviz if you want the class relationship diagrams.
- GNUplot is needed to have the… GNUplot graphs at checkpoints.

To install all those dependencies at once under Ubuntu (18.04), just type:
```bash
sudo apt install g++-8 cmake make libeigen3-dev libopenmpi-dev doxygen graphviz libgnuplot-iostream-dev.
```


### Build

The build chain uses the classical workflow of CMake.
The recommended method is to build in a specific, separated directory and call `cmake ..` from here.
CMake will prepare the compilation script for your system of choice which you can change with the `-G <generator-name>` option (see your CMake doc for the list of available generators).

Under Linux, the default is `make`, and a build command is straitghtforward:
```bash
mkdir build ; cd build ; cmake .. && make -j
```

There is, however, several build options which you may want to switch.
To see them, we recommend the use of a CMake gui, like ccmake or cmake-gui.
On the command line, you can see the available options with: `cmake -LH ..`.
Those options can be set with the `-D<option>=<value>` argument to cmake.

The first option to consider is `CMAKE_BUILD_TYPE`,
which you most probably want to set to "Debug" (during development/tests)
or "Release" (for production/validation).


### More compilation options

Other important options are: `EDO` (which is false by default)
and parallelization options: `ENABLE_OPENMP`, `MPI`, `SMP`.

By default, the build script will build the Paradiseo libraries only.

If you `ENABLE_CMAKE_TESTING` and `BUILD_TESTING`, it will build the tests,
which you can run with the `ctest` command.

If you `ENABLE_CMAKE_EXAMPLE`, it will also build the examples.

You can change the compiler used by CMake with the following options:
`CMAKE_CXX_COMPILER=/path/to/your/c++/compiler`.


Even more details
=================

Evolving Objects (EO) module
----------------------------

If you want to compile and install only the core EO module, set `EO_ONLY`,
this can be helpful if you don't need other modules with more complex dependencies.

Shared Memory Processing (SMP) module
-------------------------------------

The SMP module requires gcc 4.7 or higher. This is due to the fact that it uses the new C++ standard.

At the moment, the SMP module does not work on Windows or Mac OS X since MinGW does not provide support for std::thread and Apple does not supply a recent version of gcc (but you can try to compile gcc 4.7 by yourself).  

To enable the compilation of the SMP module, just set the `SMP` option.

Depending on your distribution, you might have to give to CMake the path of gcc and g++ 4.7.
This is the case for Ubuntu 12.04 LTS for instance.

If you are in that case and assuming you have a standard path for gcc et g++ 4.7:
```bash
cmake .. -DSMP=true -DCMAKE_C_COMPILER=/usr/bin/gcc-4.7 -DCMAKE_CXX_COMPILER=/usr/bin/g++-4.7
```

Estimating Distribution Objects (EDO) module
--------------------------------------------

To enable the compilation of the EDO module, just set the `EDO` option.

The EDO module requires a linear algebra library.
So far the core features are implemented in either [Boost::ublas](https://www.boost.org/doc/libs/release/libs/numeric/ublas) or the [Eigen3 library](https://eigen.tuxfamily.org).

The adaptive algorithms are only implemented with Eigen3, which is thus the recommended package.


Documentation
-------------

There is 2 ways to build ParadisEO documentation: module by module, or all the documentation.

Targets for the build system (usually `make`) are:
- `doc` for all documentations,
- `doc-eo` for building EO documentation,
- `doc-mo` for MO,
- `doc-edo` for MO,
- `doc-moeo` for MOEO,
- `doc-smp` for SMP.

Each documentation are generated separatly in the module build folder.
For instance, after the generation of the MO documentation, you will find it in `build/paradise-mo/doc`.

Examples
--------

Examples and lessons are generated when `ENABLE_CMAKE_EXAMPLE` is set.

If you want to build a specific lesson or example, you can check the list of available targets with `make help`.

All lessons are build on the same pattern: `<module>Lesson<number>`.
For instance, make `moLesson4` will build the Lesson 4 from the MO module. 
Easy, isn't it ?

Tests
-----

By performing tests, you can check your installation. 
Testing is disable by default, except if you build with the full install type.
To enable testing, define `ENABLE_CMAKE_TESTING` when you run cmake.

To perform tests simply type `ctest` or `make test`.

Reporting
---------

Feel free to send us reports about building, installation, tests and profiling in order to help us to improve compatibilty and installation process. Generate reports is very simple:
```bash
ctest -D Experimental
```

NOTE: Reports are anonymous, but CTest will also send informations about your configuration such as OS, CPU frequency, etc.

