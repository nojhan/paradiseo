// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "peoAggEvalFunc.h"

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

#ifndef __peoAggEvalFunc_h
#define __peoAggEvalFunc_h

#include <eoFunctor.h>

//! Interface class for creating an aggregate evaluation function.

//! The peoAggEvalFunc class offers only the interface for creating aggregate evaluation functions - there
//! are no direct internal functions provided. The class inherits <b>public eoBF< EOT&, const typename EOT :: Fitness&, void ></b>
//! thus requiring, for the derived classes, the creation of a function having the following signature:
//!
//!		<table style="border:none; border-spacing:0px;text-align:left; vertical-align:top; font-size:8pt;" border="0">
//!		<tr><td>void operator()( EOT& __eot, const typename EOT :: Fitness& __partial_fittness ); &nbsp;</td> <td> &nbsp; </td></tr>
//!		</table>
//!
//! The aggregation object is called in an iterative manner for each of the results obtained by applying partial evaluation functions.
template< class EOT > class peoAggEvalFunc : public eoBF< EOT&, const typename EOT :: Fitness&, void > {

};


#endif
