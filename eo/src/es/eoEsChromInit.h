// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoEsChromInit.h
// (c) Maarten Keijzer 2000, GeNeura Team, 1998 - EEAAX 1999
/* 
    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Contact: todos@geneura.ugr.es, http://geneura.ugr.es
             Marc.Schoenauer@polytechnique.fr
             mak@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef _eoEsChromInit_H
#define _eoEsChromInit_H

#include <es/eoEsObjectiveBounds.h>
#include <es/eoEsSimple.h>
#include <es/eoEsStdev.h>
#include <es/eoEsFull.h>
#include <utils/eoRNG.h>

#include <eoInit.h>

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

/**
\ingroup EvolutionStrategies
    
    Random Es-chromosome initializer (therefore derived from eoRnd)

    This class can initialize three types of Es's:

    eoEsSimple 
    eoEsStdev  
    eoEsFull   

    @see eoEsSimple eoEsStdev eoEsFull eoInit
*/

template <class EOT>
class eoEsChromInit : public eoInit<EOT>
{
public :
    typedef typename EOT::Fitness FitT;

    eoEsChromInit(eoEsObjectiveBounds& _bounds) : bounds(_bounds)
    {}

    EOT operator()()
    {
        return create(EOT());
    }

private :

    eoEsSimple<FitT>& create(eoEsSimple<FitT>& result)
    {
        result.resize(bounds.chromSize());

        for (unsigned i = 0; i < result.size(); ++i)
        {
            result[i] = bounds.minimum(i) + rng.uniform(bounds.maximum(i) - bounds.minimum(i));
        }

        return result;
    }

    eoEsStdev<FitT> create(eoEsStdev<FitT>& result)
    {
        unsigned chromSize = bounds.chromSize();
        result.resize(chromSize);
        result.stdevs.resize(chromSize);

        for (unsigned i = 0; i < chromSize; ++i)
        {
            double length = bounds.maximum(i) - bounds.minimum(i);
            result[i] = bounds.minimum(i) + rng.uniform(length);
        
            // Just a guess at the stdevs (anyone a better idea?)
            result.stdevs[i] = rng.uniform(length);
        }
    
        return result;
    }

    eoEsFull<FitT>  create(eoEsFull<FitT>& result)
    {
        unsigned chromSize = bounds.chromSize();
        result.resize(chromSize);
        result.stdevs.resize(chromSize);

        unsigned i;

        for (i = 0; i < chromSize; ++i)
        {
            double length = bounds.maximum(i) - bounds.minimum(i);
            result[i] = bounds.minimum(i) + rng.uniform(length);
        
            // Just a guess at the stdevs (anyone a better idea?)
            result.stdevs[i] = rng.uniform(length);
        }
        
        // nb of rotation angles: N*(N-1)/2 (in general!)
        result.correlations.resize(chromSize*(chromSize - 1) / 2);
        
        for (i = 0; i < result.correlations.size(); ++i)
        {
            // And another random guess for random initialization 
            result.correlations[i] = rng.uniform(2 * M_PI) - M_PI;
        }

        return result;
    }

    eoEsObjectiveBounds& bounds;
};

#endif