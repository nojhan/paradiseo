# ParadisEO: A C++ evolutionary computation framework to build parallel stochastic optimization solvers

## Release

The actual release is paradiseo-2.0

## Installation

The basic installation procedure must be done in a separatly folder in order to keep your file tree clean.

Create a directory to build and access it:

```
$ mkdir build && cd build
```

Compile the project into the directory with ```cmake```:
```
$ cmake ..
$ make
```

Take a coffee ;)

<font size=3>**Congratulations!! ParadiseEO is installed!**</font>

Please refer to paradisEO website or INSTALL file for further information about installation types and options.

---

## Directory Structure

* __AUTHORS__:    Authors list;

* __cmake__:  Directory of cmake files;

* __CMakeLists.txt__: Definitions for building process;

* __CTestConfig.cmake__:  Definitions for testing process;

* __INSTALL__:    Steps and options of the installation process;

* __LICENSE__: License contents;

* __eo__: Specific directory of the EO (Evolving Objects) module;

* __mo__: Specific directory of the MO (Moving Objects) module;

* __moeo__: Specific directory of the MOEO (Multi-Objective Optimization) module;

* __problems__: classical problems evaluation functions.

---
## Notes

ParadisEO uses EO, a templates-based, ANSI-C++ compliant evolutionary computation library. It contains classes for almost any kind of evolutionary computation you might come up to - at least for the ones we could think of. 

EO Website: http://eodev.sourceforge.net/.

ParadisEO Website: http://paradiseo.gforge.inria.fr/

EO is distributed under the GNU Lesser General Public License: http://www.gnu.org/copyleft/lesser.html

Please read README file of each extracted directory if you have problems for the installation.

---
<font size=3>**Contact**</font>

<font size=2>For further information about ParadisEO, help or to report any problem : paradiseo-help@lists.gforge.inria.fr</font>