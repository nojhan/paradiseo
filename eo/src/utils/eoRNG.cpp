#include <ctime>
#include "eoRNG.h"

namespace eo
{
/// The Global random number generator. 
eoRng rng((uint32) time(0));
}

