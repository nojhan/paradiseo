This package contains the source code for DTLZ problems.

# Step 1 - Configuration
------------------------
Edit the "install.cmake" file by entering the FULL PATH of the "ParadisEO".
On Windows write your path with double antislash (ex: C:\\Users\\...)


# Step 2 - Build process
------------------------
ParadisEO is assumed to be compiled. To download ParadisEO, please visit http://paradiseo.gforge.inria.fr/.
Go to the DLTZ/build/ directory and lunch cmake:
(Unix)       > cmake ..
(Windows)    > cmake .. -G"Visual Studio 9 2008"

Note for windows users: if you don't use VisualStudio 9, enter the name of your generator instead of "VisualStudio 9 2008".


# Step 3 - Compilation
----------------------
In the DTLZ/build/ directory:
(Unix)       > make
(Windows)    Open the VisualStudio solution and compile it (Windows).
You can refer to this tutorial if you don't know how to compile a solution: http://paradiseo.gforge.inria.fr/index.php?n=Paradiseo.VisualCTutorial


# Step 4 - Execution
---------------------
A toy example is given to test the components. You can run these tests as following.
To define problem-related components for your own problem, please refer to the tutorials available on the website : http://paradiseo.gforge.inria.fr/.
In the DTLZ/build/ directory:
(Unix)       > ctest
Windows users, please refer to this tutorial: http://paradiseo.gforge.inria.fr/index.php?n=Paradiseo.VisualCTutorial

In the directory "application", there are three ".cpp" which instantiate IBEA, NSGAII and SPEA2 on ZDT problems. To change of algorithms, you can compare these three files and see the few changes to do.

(Unix) After compilation you can run the script "DTLZ/run.sh" and see results in "IBEA.out", "NSGAII.out" and "SPEA2.out". Parameters can be modified in the script.

(Windows) Add argument "IBEA.param", "SPEA2.param" or "NSGAII.param" and execute the corresponding algorithms.
Windows users, please refer to this tutorial: http://paradiseo.gforge.inria.fr/index.php?n=Paradiseo.VisualCTutorial

# Documentation
---------------
The API-documentation is available in doc/html/index.html 

