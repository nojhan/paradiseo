/*
* <peoSeqPopEval.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Sebastien Cahon, Alexandru-Adrian Tantar
*
* This software is governed by the CeCILL license under French law and
* abiding by the rules of distribution of free software.  You can  use,
* modify and/ or redistribute the software under the terms of the CeCILL
* license as circulated by CEA, CNRS and INRIA at the following URL
* "http://www.cecill.info".
*
* As a counterpart to the access to the source code and  rights to copy,
* modify and redistribute granted by the license, users are provided only
* with a limited warranty  and the software's author,  the holder of the
* economic rights,  and the successive licensors  have only  limited liability.
*
* In this respect, the user's attention is drawn to the risks associated
* with loading,  using,  modifying and/or developing or reproducing the
* software by the user in light of its specific status of free software,
* that may mean  that it is complicated to manipulate,  and  that  also
* therefore means  that it is reserved for developers  and  experienced
* professionals having in-depth computer knowledge. Users are therefore
* encouraged to load and test the software's suitability as regards their
* requirements in conditions enabling the security of their systems and/or
* data to be ensured and,  more generally, to use and operate it in the
* same conditions as regards security.
* The fact that you are presently reading this means that you have had
* knowledge of the CeCILL license and that you accept its terms.
*
* ParadisEO WebSite : http://paradiseo.gforge.inria.fr
* Contact: paradiseo-help@lists.gforge.inria.fr
*
*/

#ifndef __peoSeqPopEval_h
#define __peoSeqPopEval_h

#include <eoEvalFunc.h>

#include "peoPopEval.h"

//! Sequential evaluation functor wrapper.

//! The peoSeqPopEval class acts only as a ParadisEO specific sequential evaluation functor - a wrapper for incorporating
//! an <b>eoEvalFunc< EOT ></b>-derived class as evaluation functor. The specified EO evaluation object is applyied in an
//! iterative manner to each individual of a specified population.
template< class EOT > class peoSeqPopEval : public peoPopEval< EOT >
  {

  public:

    //! Constructor function - it only sets an internal reference to point to the specified evaluation object.
    //!
    //! @param eoEvalFunc< EOT >& __eval - evaluation object to be applied for each individual of a specified population
    peoSeqPopEval( eoEvalFunc< EOT >& __eval );

    //! Operator for evaluating all the individuals of a given population - in a sequential iterative manner.
    //!
    //! @param eoPop< EOT >& __pop - population to be evaluated.
    void operator()( eoPop< EOT >& __pop );
    void operator()( eoPop< EOT > &__dummy,eoPop< EOT >&__pop);

  private:

    eoEvalFunc< EOT >& eval;
  };


template< class EOT > peoSeqPopEval< EOT > :: peoSeqPopEval( eoEvalFunc< EOT >& __eval ) : eval( __eval )
{}

template< class EOT > void peoSeqPopEval< EOT > :: operator()( eoPop< EOT >& __dummy,eoPop< EOT >& __pop )
{
	this->operator()(__pop);
}

template< class EOT > void peoSeqPopEval< EOT > :: operator()( eoPop< EOT >& __pop )
{

  for ( unsigned i = 0; i < __pop.size(); i++ )
    eval( __pop[i] );
}


#endif
