/*
* <peoAggEvalFunc.h>
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
template< class EOT > class peoAggEvalFunc : public eoBF< EOT&, const typename EOT :: Fitness&, void >
  {};


#endif
