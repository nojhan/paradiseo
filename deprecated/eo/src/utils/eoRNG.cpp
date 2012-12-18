#ifdef _MSC_VER
// to avoid long name warnings
#pragma warning(disable:4786)
#endif

#include <ctime>
#include "eoRNG.h"

// initialize static constants
const uint32_t eoRng::K(0x9908B0DFU);
const int eoRng::M(397);
const int eoRng::N(624);

namespace eo
{
    // global random number generator object
    eoRng rng(static_cast<uint32_t>(time(0)));
}
