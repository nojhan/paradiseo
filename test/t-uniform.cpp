#include <eo>
#include <do>

#include "Rosenbrock.h"

typedef eoReal< eoMinimizingFitness > EOT;

int main(void)
{
    eoState state;

    doUniform< EOT >* distrib = new doUniform< EOT >( EOT(3, -1), EOT(3, 1) );
    state.storeFunctor(distrib);

    return 0;
}
