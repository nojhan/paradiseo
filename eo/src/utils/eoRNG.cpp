#include <ctime>
#include "eoRNG.h"

#include <sys/time.h>
/** this function retunrs a "truly random" seed for RNG
 * Replaces the call to time(0) that seems to be non-portable???
 */
uint32 random_seed()
{
   struct timeval tval;
   struct timezone tzp;

   gettimeofday (&tval, &tzp);	// milliseconds since midnight January 1, 1970.
   return uint32(tval.tv_usec);
}

namespace eo
{
/// The Global random number generator. 
eoRng rng(random_seed());
}

