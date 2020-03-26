#include <iostream>

#include <eo>
#include <es.h>

using namespace std;

int main(int /*argc*/, char** /*argv[]*/)
{
    typedef eoReal<eoMinimizingFitness> EOT;

    // Build something like: ">&1 echo 1.2; # INVALID 2 1 2"
    eoEvalCmd<EOT> eval("echo 1.2", ">&1", "; #");

    EOT sol = {1,2};
    std::clog << sol << std::endl;

    try {
        eval(sol);
    } catch(eoSystemError& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }

    std::clog << eval.last_call() << std::endl;
    EOT::Fitness f = sol.fitness();
    std::clog << "fitness: " << f << std::endl;
    assert(f = 1.2);
}
