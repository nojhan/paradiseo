// (c) Thales group, 2010
/*
    Authors:
             Johann Dreo <johann.dreo@thalesgroup.com>
*/

#ifndef _doBounderUniform_h
#define _doBounderUniform_h

#include "doBounder.h"

template < typename EOT >
class doBounderUniform : public doBounder< EOT >
{
public:
    doBounderUniform( EOT min, EOT max )
	: doBounder< EOT >( min, max )
    {}

    void operator()( EOT& sol )
    {
        unsigned int size = sol.size();
        assert(size > 0);

        for (unsigned int d = 0; d < size; ++d) {

            if ( sol[d] < this->min()[d] || sol[d] > this->max()[d]) {
                // use EO's global "rng"
                sol[d] = rng.uniform( this->min()[d], this->max()[d] );
            }
        } // for d in size
    }
};

#endif // !_doBounderUniform_h
