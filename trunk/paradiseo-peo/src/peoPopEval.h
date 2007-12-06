/*
* <peoPopEval.h>
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

#ifndef __peoPopEval_h
#define __peoPopEval_h

#include "core/service.h"

//! Interface for ParadisEO specific evaluation functors.

//! The <b>peoPopEval</b> class provides the interface for constructing ParadisEO specific evaluation functors.
//! The derived classes may be used as wrappers for <b>EO</b>-derived evaluation functors. In order to have an example,
//! please refer to the implementation of the <b>peoSeqPopEval</b> and <b>peoParaPopEval</b> classes.
template< class EOT > class peoPopEval : public Service, public eoPopEvalFunc<EOT>
{

public:

  //! Interface function providing the signature for constructing an evaluation functor.
  virtual void operator()( eoPop< EOT >& __tmp, eoPop< EOT >& __pop )=0;
  
};

#endif
