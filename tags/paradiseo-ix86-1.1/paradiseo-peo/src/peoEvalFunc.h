/*
* <peoEvalFunc.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2008
* (C) OPAC Team, INRIA, 2008
*
* Clive Canape
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
* Contact: clive.canape@inria.fr
*
*/

#ifndef PEOEVALFUNC_H
#define PEOEVALFUNC_H

//! @class peoEvalFunc
//! @brief Specific class for evaluation
//! @see eoEvalFunc
//! @version 1.0
//! @date november 2007
#ifdef _MSC_VER
template< class EOT, class FitT = EOT::Fitness, class FunctionArg = const EOT& >
#else
template< class EOT, class FitT = typename EOT::Fitness, class FunctionArg = const EOT& >
#endif
struct peoEvalFunc: public eoEvalFunc<EOT>
  {

    //! @brief Constructor
    //! @param FitT (* _eval)( FunctionArg )
    peoEvalFunc( FitT (* _eval)( FunctionArg ) )
        : eoEvalFunc<EOT>(), evalFunc( _eval )
    {};

    //! @brief Virtual operator
    //! @param EOT & _peo
    virtual void operator() ( EOT & _peo )
    {
      _peo.fitness((*evalFunc)( _peo ));
    };

private:
    //! @param FitT (* evalFunc )( FunctionArg )
    FitT (* evalFunc )( FunctionArg );
  };

#endif

