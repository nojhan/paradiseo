/* 
* <peoNoAggEvalFunc.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Sébastien Cahon, Alexandru-Adrian Tantar
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

#ifndef __peoNoAggEvalFunc_h
#define __peoNoAggEvalFunc_h

#include "peoAggEvalFunc.h"

//! Class providing a simple interface for associating a fitness value to a specified individual.

//! The peoNoAggEvalFunc class does nothing more than an association between a fitness value and a specified individual.
//! The class is provided as a mean of declaring that no aggregation is required for the evaluation function - the fitness
//! value is explicitly specified.
template< class EOT > class peoNoAggEvalFunc : public peoAggEvalFunc< EOT > {

public :

	//! Operator which sets as fitness the <b>__fit</b> value for the <b>__sol</b> individual
	void operator()( EOT& __sol, const typename EOT :: Fitness& __fit );
};


template< class EOT > void peoNoAggEvalFunc< EOT > :: operator()( EOT& __sol, const typename EOT :: Fitness& __fit ) {

	__sol.fitness( __fit );
}


#endif
