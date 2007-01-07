// "peoAggEvalFunc.h"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
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
