#ifdef _MSC_VER
// to avoid long name warnings
#pragma warning(disable:4786)
#endif 

#include <ctime>
#include "eoRNG.h"

namespace eo
{
/// The Global random number generator. 
eoRng rng(time(0));
}

