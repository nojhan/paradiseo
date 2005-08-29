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

#include <cmath>
#include <es/eoRealInitBounded.h>
#include <es/eoEsSimple.h>
#include <es/eoEsStdev.h>
#include <es/eoEsFull.h>

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

/**
\ingroup EvolutionStrategies

    Random Es-chromosome initializer (therefore derived from eoInit)

    This class can initialize four types of real-valued genotypes
    thanks to tempate specialization of private method create

    eoReal          just an eoVector<double>
    eoEsSimple      + one self-adapting single sigma for all variables
    eoEsStdev       a whole std::vector of self-adapting sigmas
    eoEsFull        a full self-adapting correlation matrix

    @see eoReal eoEsSimple eoEsStdev eoEsFull eoInit
*/

template <class EOT>
class eoEsChromInit : public eoRealInitBounded<EOT>
{
public:

    using eoEsChromInit< EOT >::size;
    using eoEsChromInit< EOT >::theBounds;

    typedef typename EOT::Fitness FitT;

    /** Ctor:

    @param eoRealVectorBounds& _bounds : bounds for uniform initialization
    @param _sigma : initial value for the stddev
    @param _to_scale : wether sigma should be multiplied by the range of each variable
                       added December 2004 - MS (together with the whole comment :-)
    */
    eoEsChromInit(eoRealVectorBounds& _bounds, double _sigma = 0.3, bool _to_scale=false)
        : eoRealInitBounded<EOT>(_bounds)
    {
    // a bit of pre-computations, to save time later (even if some are useless)

    // first, the case of one unique sigma
    if (_to_scale)   // sigma is scaled by the average range (if that means anything!)
      {
	double scaleUnique = 0;
	for (unsigned i=0; i<size(); i++)
	  scaleUnique += theBounds().range(i);
	scaleUnique /= size();
	uniqueSigma = _sigma * scaleUnique;
      }
    else
      uniqueSigma = _sigma;

    // now the case of a vector of sigmas
    // first allocate
    lesSigmas.resize(size());	   // size() is the size of the bounds (see eoRealInitBounded)

    for (unsigned i=0; i<size(); i++)
      if (_to_scale)   // each sigma is scaled by the range of the corresponding variable
	{
	  lesSigmas[i] = _sigma * theBounds().range(i);
	}
      else
	lesSigmas[i] = _sigma;
    }

  void operator()(EOT& _eo)
  {
    eoRealInitBounded<EOT>::operator()(_eo);
    create_self_adapt(_eo);
    _eo.invalidate();		   // was MISSING!!!!
  }

  // accessor to sigma
  //  double sigmaInit() {return sigma;}

private :

  // No adaptive mutation at all
  void create_self_adapt(eoReal<FitT>&)// nothing to do here ...
  { }

  // Adaptive mutation through a unique sigma
  void create_self_adapt(eoEsSimple<FitT>& result)
  {
    // pre-computed in the Ctor
      result.stdev = uniqueSigma;
  }

  // Adaptive mutation through a std::vector of sigmas
  void create_self_adapt(eoEsStdev<FitT>& result)
  {
    result.stdevs = lesSigmas;
  }

  // Adaptive mutation through a whole correlation matrix
  void create_self_adapt(eoEsFull<FitT>& result)
  {
    // first the stdevs (pre-computed in the Ctor)
    result.stdevs = lesSigmas;
    unsigned int theSize = size();
    // nb of rotation angles: N*(N-1)/2 (in general!)
    result.correlations.resize(theSize*(theSize - 1) / 2);
    for (unsigned i=0; i<result.correlations.size(); ++i)
      {
	// uniform in [-PI, PI)
	result.correlations[i] = rng.uniform(2 * M_PI) - M_PI;
      }
  }

  // the DATA
  double uniqueSigma;		   // initial value in case of a unique sigma
  std::vector<double> lesSigmas;  // initial values in case of a vector fo sigmas
};

#endif
