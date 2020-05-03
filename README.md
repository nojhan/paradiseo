# Paradiseo: a Heuristic Optimization Framework

Paradiseo is an open-source ***full-featured evolutionary computation framework*** which main purpose is to help you write ***your own stochastic optimization algorithms***, insanely fast.

It focus on the efficiency of the implementation of solvers, by providing:
- a ***modular design*** for several types of paradigms,
- the ***largest codebase*** of existing components,
- tools for ***automated design and selection*** of algorithms,
- a focus on ***speed*** and several ***parallelization*** options.

![Paradiseo logo](https://github.com/nojhan/paradiseo/blob/master/website/paradiseo_logo_200px_dark.png)

# Rationale

## Black-box and Gray-box Optimization Problems

Paradiseo targets the development of solvers for mathematical optimization
problems for which you cannot compute gradients.
The classical use case is the automated design or configuration of
some system which is simulated.

## Metaheuristics / Evolutionary Algorithms

Paradiseo targets the design of metaheuristics solvers using
computational intelligence methods, a subdomain of artificial intelligence.

## Why choosing Paradiseo?

Learning a full-featured framework like Paradiseo very often seems overkill.
However, we would like to stress out that you may forget some points
while jumping to this conclusion.

> **Paradiseo provides the *largest mature codebase* of state-of-the-art algorithms, and is focused on (automatically) find the *most efficient solvers*.**

The most classical impediment to the use of Paradiseo is that you just want to check if your problem can actually be solved with heuristics. You feel that it would be a loss of time to learn complex stuff if it ends being useless.

However, you should keep in mind that:

- Metaheuristics do seem very easy to implement in textbooks, but the state-of-the art versions of efficient algorithms can be a lot more complex.
- It is usually easy to get something to actually run, but it is far more difficult to get an efficient solver.
- Metaheuristics performances on a given problem are very sensitive to small variations in the parameter setting or the choice of some operators. Which render large experimental plans and algorithm selection compulsory to attain peak efficiency.

> **Fortunately, Paradiseo have the *largest codebase* of the market, hardened along 20 years of development of tens of solvers. Additionally, it provides the tools to rapidly search for the best combination of algorithms to solve your problem, even searching for this combination *automatically*.**

> Paradiseo is the fastest framework on the market, which is a crucial feature for modern and robust approach to solver design and validation.

Another classical criticism against Paradiseo is that C++ is hard and that a fast language is useless because speed is not a concern when your objective function is dominating all the runtime.

However, we argue that:

- During the design phase of your solver, you will need to estimate its performance against synthetic benchmarks that are fast to compute. In that case, fast computation means fast design iterations. And it's even more true if you plan to use automated design to find the best solver for your problem.
- Modern C++ makes use of the very same high-level abstractions you would find in more accepted languages like Python. Sure, the syntax is cumbersome, but you will not see it after a while, given that you will work at the algorithm level.
- C++ provides full type checking and the largest set of tooling for any modern language, which are your first line of defense against long-term bugs. Sure, it sometimes gives you the impression that you fight against the compiler, but chasing subtle interface bugs across a complex Python code is even harder.

# Features

## Component-based Design

Designing an algorithm with Paradiseo consists in choosing what components (called operators) you want to use for your specific needs, just as building a structure with Lego blocks.

If you have a classical problem for which available code exists (for example if you have a black-box problem with real-valued variables), you will just choose operators to form an algorithm and connect it to your evaluation function (which computes the quality of a given solution).

If your problem is a bit more exotic, you will have to code a class that encodes how solutions to your problem are represented, and perhaps a few more. For instance, you may want ad-hoc variations operators, but most of the other operators (selection, replacement, stopping criteria, command-line interface, etc.) are already available in Paradiseo.

## Large Choice of Components

Paradiseo is organized in several modules, either providing different "grammars" for different algorithms, either providing high-level features. All modules follows the same architecture design and are interoperable with the others, so that you can easily choose the subset of features you need.

It is, for instance, easy to start with a simple local search, then add multi-objective capabilities, then shared-memory parallelization, then hybridization with an evolutionary algorithm and finally plug everything in an objective function so as to optimize the parameters with a particle swarm optimizer.

## Portability

Paradiseo is mainly developed under Linux operating systems, where its dependencies and the C++ toolchain are easy to install. Recent versions have been tested with gcc and clang compilers.

Stable versions should however work on Windows and any Unix-like operating system with a standard-conforming C++ development system. 


# Code

The latest stable version is on the official Git repository of INRIA: `git clone git://scm.gforge.inria.fr/paradiseo/paradiseo.git`

## Dependencies

In order to build the latest version of Paradiseo, you will need a C++ compiler supporting C++17. So far, GCC and CLANG gave good results under Linux. You will also need the CMake and make build tools.

A free working build chain under Windows seems always difficult to find. Paradiseo 2.0.1 was successfully tested with MinGW (minus the PEO module), but it's unsure if it still work for recent versions. If you managed to build under Windows, your feedback would be appreciated.

Some features are only available if some dependencies are installed:

- Most of the EDO module depends on either uBlas or Eigen3. The recommended package is Eigen3, which enables the adaptive algorithms.
- Doxygen is needed to build the API documentation, and you should also install graphviz if you want the class relationship diagrams.
- GNUplot is needed to have the… GNUplot graphs at checkpoints.

> To install all those dependencies at once under Ubuntu (18.04), just type: `sudo apt install g++-8 cmake make libeigen3-dev libopenmpi-dev doxygen graphviz libgnuplot-iostream-dev`.

## Compilation

The build chain uses the classical workflow of CMake. The recommended method is to build in a specific, separated directory and call `cmake ..` from here. CMake will prepare the compilation script for your system of choice which you can change with the `-G <generator-name>` option (see the CMake doc for the list of available generators).

Under Linux, the default is make, and a build command is straitghtforward: `mkdir build ; cd build ; cmake .. && make -j`.

There is, however, several build options which you may want to switch. To see them, we recommend the use of a CMake gui, like ccmake or cmake-gui . On the command line, you can see the available options with: `cmake -LH ..` . Those options can be set with the `-D<option>=<value>` argument to cmake.

**The first option to consider is `CMAKE_BUILD_TYPE`, which you most probably want to set to `Debug` (during development/tests) or `Release` (for production/validation).**


Other important options are: `EDO` (which is false by default) and parallelization options: `ENABLE_OPENMP`, `MPI`, `SMP`.

By default, the build script will build the Paradiseo libraries only.

If you `ENABLE_CMAKE_TESTING` and `BUILD_TESTING`, it will be the tests, which you can run with the `ctest` command.

If you `ENABLE_CMAKE_EXAMPLE`, it will also build the examples.

## Licenses

Paradiseo is distributed under the GNU Lesser General Public License and the CeCILL license (depending on the modules).

Note that those licenses places copyleft restrictions on a program created with Paradiseo, but does not apply these restrictions to other software that would links with the program.


# Documentation

Paradiseo has a lot of documentation! You will find in the source repository
a lot of examples, the tutorials and you can generate the API documentation
(`make doc`, then open
`paradiseo/<build>/<module>/doc/html/index.html`).

Tutorials are located in each module's directory. For example for the EO module:
`paradiseo/eo/tutorial`.
A lot of examples for (almost) each class are available in the test directories
(e.g. `paradiseo/eo/test`). Example problems and bindings to external
benchmark libraries are in `paradiseo/problems`.

For academic articles, books, more tutorials, presentations slides,
real life example of solvers and contact information,
please see the web site (available in `paradiseo/website/index.html`).

