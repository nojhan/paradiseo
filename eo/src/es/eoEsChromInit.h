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
    eoEsStdev       a whole vector of self-adapting sigmas
    eoEsFull        a full self-adapting correlation matrix

    @see eoReal eoEsSimple eoEsStdev eoEsFull eoInit
*/

template <class EOT>
class eoEsChromInit : public eoRealInitBounded<EOT>
{
public :
  typedef typename EOT::Fitness FitT;

  eoEsChromInit(eoRealVectorBounds& _bounds, double _sigma = 0.3) : 
    eoRealInitBounded<EOT>(_bounds), sigma(_sigma)  {}

  void operator()(EOT& _eo)
  {
    eoRealInitBounded<EOT>::operator()(_eo);
    create_self_adapt(_eo);
    _eo.invalidate();		   // was MISSING!!!!
  }

  // accessor to sigma
  double sigmaInit() {return sigma;}

private :

  // No adaptive mutation at all
  void create_self_adapt(eoReal<FitT>& result)// nothing to do here ...
  { }

  // Adaptive mutation through a unique sigma
  void create_self_adapt(eoEsSimple<FitT>& result)
  {
    // sigma is scaled by the average range (if that means anything!)
    result.stdev = sigma;
  }

  // Adaptive mutation through a vector of sigmas
  void create_self_adapt(eoEsStdev<FitT>& result)
  {
    unsigned theSize = eoRealInitBounded<EOT>::size();
    result.stdevs.resize(theSize);
    for (unsigned i = 0; i < theSize; ++i)
      {
	// should we scale sigmas to the corresponding object variable range?
	result.stdevs[i] = sigma;
      }
  }

  // Adaptive mutation through a whole correlation matrix
  void create_self_adapt(eoEsFull<FitT>& result)
  {
    unsigned theSize = eoRealInitBounded<EOT>::size();

    result.stdevs.resize(theSize);
    for (unsigned i = 0; i < theSize; ++i)
      {
	// should we scale sigmas to the corresponding object variable range?
	result.stdevs[i] = sigma;
      }
        
    // nb of rotation angles: N*(N-1)/2 (in general!)
    result.correlations.resize(theSize*(theSize - 1) / 2);
    for (unsigned i = 0; i < result.correlations.size(); ++i)
      {
	// uniform in [-PI, PI)
	result.correlations[i] = rng.uniform(2 * M_PI) - M_PI;
      }
  }

  double sigma;			   // initial value for sigmas
};

#endif
