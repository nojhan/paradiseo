//
/* (c) Maarten Keijzer 2000, GeNeura Team, 1998 - EEAAX 1999

This library is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the Free
Software Foundation; either version 2 of the License, or (at your option) any
later version.

This library is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with this library; if not, write to the Free Software Foundation, Inc., 59
Temple Place, Suite 330, Boston, MA 02111-1307 USA

Contact: http://eodev.sourceforge.net
         todos@geneura.ugr.es, http://geneura.ugr.es
         Marc.Schoenauer@polytechnique.fr
         mak@dhi.dk
 */


#ifndef _eoEsChromInit_H
#define _eoEsChromInit_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

#include <es/eoRealInitBounded.h>
#include <es/eoEsSimple.h>
#include <es/eoEsStdev.h>
#include <es/eoEsFull.h>

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

/** Random Es-chromosome initializer (therefore derived from eoInit)

@ingroup Real
@ingroup Initializators


This class can initialize four types of real-valued genotypes thanks
to tempate specialization of private method create:

- eoReal          just an eoVector<double>
- eoEsSimple      + one self-adapting single sigma for all variables
- eoEsStdev       a whole std::vector of self-adapting sigmas
- eoEsFull        a full self-adapting correlation matrix

@see eoReal eoEsSimple eoEsStdev eoEsFull eoInit
*/
template <class EOT>
class eoEsChromInit : public eoRealInitBounded<EOT>
{
public:

    using eoRealInitBounded<EOT>::size;
    using eoRealInitBounded<EOT>::theBounds;

    typedef typename EOT::Fitness FitT;

    /** Constructor

    @param _bounds bounds for uniform initialization
    @param _sigma initial value for the stddev
    @param _to_scale wether sigma should be multiplied by the range of each variable
    added December 2004 - MS (together with the whole comment :-)
    */
    eoEsChromInit(eoRealVectorBounds& _bounds, double _sigma = 0.3, bool _to_scale=false)
        : eoRealInitBounded<EOT>(_bounds)
        {
            // a bit of pre-computations, to save time later (even if some are useless)
            //
            // first, in the case of one unique sigma
            // sigma is scaled by the average range (if that means anything!)
            if (_to_scale)
            {
                double scaleUnique = 0;
                for (unsigned i=0; i<size(); i++)
                    scaleUnique += theBounds().range(i);
                scaleUnique /= size();
                uniqueSigma = _sigma * scaleUnique;
            }
            else
                uniqueSigma = _sigma;
            // now the case of a vector of sigmas first allocate space according
            // to the size of the bounds (see eoRealInitBounded)
            vecSigma.resize(size());
            // each sigma is scaled by the range of the corresponding variable
            for(unsigned i=0; i<size(); i++)
                if(_to_scale)
                    vecSigma[i] = _sigma * theBounds().range(i);
                else
                    vecSigma[i] = _sigma;
        }


    /** Constructor

    @overload

    Specify individual initial sigmas for each variable.

    @param _bounds bounds for uniform initialization
    @param _vecSigma initial value for the stddev
    */
    eoEsChromInit(eoRealVectorBounds& _bounds, const std::vector<double>& _vecSigma)
        : eoRealInitBounded<EOT>(_bounds), uniqueSigma(_vecSigma[0]), vecSigma(_vecSigma)
        {
            assert(_bounds.size() == size());
            assert(_vecSigma.size() == size());
        }


    void operator()(EOT& _eo)
        {
            eoRealInitBounded<EOT>::operator()(_eo);
            create_self_adapt(_eo);
            _eo.invalidate();
        }


private:

    /** Create intializer

    No adaptive mutation at all
    */
    void create_self_adapt(eoReal<FitT>&)
        {}



    /** Create intializer

    @overload

    Adaptive mutation through a unique sigma
    */
    void create_self_adapt(eoEsSimple<FitT>& result)
        {
            // pre-computed in the Ctor
            result.stdev = uniqueSigma;
        }



    /** Create intializer

    @overload

    Adaptive mutation through a std::vector of sigmas

    @todo Should we scale sigmas to the corresponding object variable range?
    */
    void create_self_adapt(eoEsStdev<FitT>& result)
        {
            // pre-computed in the constructor
            result.stdevs = vecSigma;
        }



    /** Create intializer

    @overload

    Adaptive mutation through a whole correlation matrix
    */
    void create_self_adapt(eoEsFull<FitT>& result)
        {
            // first the stdevs (pre-computed in the Ctor)
            result.stdevs = vecSigma;
            unsigned int theSize = size();
            // nb of rotation angles: N*(N-1)/2 (in general!)
            result.correlations.resize(theSize*(theSize - 1) / 2);
            for (unsigned i=0; i<result.correlations.size(); ++i)
            {
                // uniform in [-PI, PI)
                result.correlations[i] = rng.uniform(2 * M_PI) - M_PI;
            }
        }



    /** Initial value in case of a unique sigma */
    double uniqueSigma;

    /** Initial values in case of a vector of sigmas */
    std::vector<double> vecSigma;
};

#endif



// Local Variables:
// coding: iso-8859-1
// mode:C++
// c-file-style: "Stroustrup"
// comment-column: 35
// fill-column: 80
// End:
