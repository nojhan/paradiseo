#ifndef _moIntensification_h
#define _moIntensification_h

#include <memory/moMemory.h>
#include <eoFunctor.h>

/**
 * Abstract class for intensification strategy
 */
template< class Neighbor >
class moIntensification : public moMemory<Neighbor>, public eoUF<typename Neighbor::EOT &,bool>
{};

#endif
