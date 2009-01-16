/*
 <moExpl.h>
 Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2008
 (C) OPAC Team, LIFL, 2002-2008

 Salma Mesmoudi (salma.mesmoudi@inria.fr), Jean-Charles Boisson (Jean-Charles.Boisson@lifl.fr)

 This software is governed by the CeCILL license under French law and
 abiding by the rules of distribution of free software.  You can  use,
 modify and/ or redistribute the software under the terms of the CeCILL
 license as circulated by CEA, CNRS and INRIA at the following URL
 "http://www.cecill.info".

 As a counterpart to the access to the source code and  rights to copy,
 modify and redistribute granted by the license, users are provided only
 with a limited warranty  and the software's author,  the holder of the
 economic rights,  and the successive licensors  have only  limited liability.

 In this respect, the user's attention is drawn to the risks associated
 with loading,  using,  modifying and/or developing or reproducing the
 software by the user in light of its specific status of free software,
 that may mean  that it is complicated to manipulate,  and  that  also
 therefore means  that it is reserved for developers  and  experienced
 professionals having in-depth computer knowledge. Users are therefore
 encouraged to load and test the software's suitability as regards their
 requirements in conditions enabling the security of their systems and/or
 data to be ensured and,  more generally, to use and operate it in the
 same conditions as regards security.
 The fact that you are presently reading this means that you have had
 knowledge of the CeCILL license and that you accept its terms.

 ParadisEO WebSite : http://paradiseo.gforge.inria.fr
 Contact: paradiseo-help@lists.gforge.inria.fr
 */
#ifndef _moExpl_h
#define _moExpl_h

#include <eoFunctor.h>

//! Description of an explorer
/*!
  Only a description...See moMoveLoopExpl.
*/
template < class EOT >
class moExpl : public eoBF < const EOT &, EOT &, bool >
{
public:
	unsigned int i;

	//Neighborhoods vector
	std::vector< eoMonOp<EOT>* > explore;
	
	//! Generic constructor
  /*!
    Generic constructor using a eoMonOp

    \param _expl Algorithme or mutation.
    
  */
	moExpl(eoMonOp<EOT> & expl){
	    i=0;
	    add(expl);
	}
	
	//! Generic constructor
  /*!
    Generic constructor using a eoMonOp

    \param _expl Algorithme or mutation.
    
  */
	
	//!  Procedure which launches the moExpl.
  /*!
    The exploration starts from an old solution and provides a new solution.

    \param _old_solution The current solution.
    \param _new_solution The new solution (result of the procedure).
  */

	bool operator ()(const EOT & _old, EOT & _new){
	    _new=(EOT)_old;
	    return (*explore[i])(_new);
	}
	//add an algorithm or mutation to neighborhoods vector
	void add(eoMonOp<EOT> & expl){
		explore.push_back(&expl);
	}
	//setIndice make sur that the initial indice (_i) is not bigger than the explorer size.
	void setIndice(unsigned int _i){
		if( _i >= explore.size() ){
		  std::cout << "[" << _i << "]" << std::endl;
		  throw std::runtime_error("[moExpl.h]: bad index "+_i);
		}
		i=_i;
	}

	//return the size of the class
	unsigned int size(){
		return explore.size();
	}	

};

#endif
