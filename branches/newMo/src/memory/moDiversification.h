#ifndef _moDiversification_h
#define _moDiversification_h

#include <memory/moMemory.h>

/**
 * Abstract class for diversification strategy
 */
template< class Neighbor >
class moDiversification : public moMemory<Neighbor>, public eoUF<typename Neighbor::EOT &,bool>
{};

#endif
