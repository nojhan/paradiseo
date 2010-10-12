#include <eo>
#include <do>

#include "Rosenbrock.h"

typedef eoReal< eoMinimizingFitness > EOT;
typedef doUniform< EOT > Distrib;

int main(void)
{
    eoState state;

    doContinue< Distrib >* continuator = new doDummyContinue< Distrib >();
    state.storeFunctor(continuator);

    return 0;
}
