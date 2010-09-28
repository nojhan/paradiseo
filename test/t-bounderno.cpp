#include <eo>
#include <do>
#include <mo>

#include <utils/eoLogger.h>
#include <utils/eoParserLogger.h>

#include "Rosenbrock.h"
#include "Sphere.h"

typedef eoReal< eoMinimizingFitness > EOT;

int main(void)
{
    doBounderNo< EOT > bounder;

    return 0;
}
