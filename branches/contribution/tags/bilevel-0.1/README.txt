Paradiseo contribution for a bilevel multiple depot vehicule routing problem solver

Build instructions:
start by setting the path to paradiseo in install.cmake (paradiseo should be built)
then go to the build directory, and do a "cmake ..", then a "make"
after building, in the build directory

Run instructions:
./application/simpletemp bench seed
where bench is a benchmark file for biMDVRP (bipr*x* in the instances folder) and seed a number to initialize the pseudo random generator
./application/multitemp bench seed
where bench is a benchmark file for mbiMDVRP (mbipr*x* in the instances folder) and seed a number to initialize the pseudo random generator



Additional informations:
*The instances/benchmark are in instances directory
*We use CoBRA to solve the problem
