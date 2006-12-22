// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "peoPopEval.h"

// (c) OPAC Team, LIFL, August 2005

/* This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2 of the License, or (at your option) any later version.
   
   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.
   
   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
   
   Contact: cahon@lifl.fr
*/

#ifndef __peoPopEval_h
#define __peoPopEval_h

#include "core/service.h"

//! Interface for ParadisEO specific evaluation functors.

//! The <b>peoPopEval</b> class provides the interface for constructing ParadisEO specific evaluation functors.
//! The derived classes may be used as wrappers for <b>EO</b>-derived evaluation functors. In order to have an example,
//! please refer to the implementation of the <b>peoSeqPopEval</b> and <b>peoParaPopEval</b> classes.
template< class EOT > class peoPopEval : public Service {

public:

	//! Interface function providing the signature for constructing an evaluation functor.
	virtual void operator()( eoPop< EOT >& __pop ) = 0;
};


#endif
