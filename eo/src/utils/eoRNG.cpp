#include <ctime>
#include "eoRNG.h"

/// The global object, should probably be initialized with an xor
/// between time and process_id.
eoRng rng((uint32) time(0));

