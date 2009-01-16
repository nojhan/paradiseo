/*
 <moVNS.h>
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

#ifndef _moVNS_h
#define _moVNS_h

#include <eoEvalFunc.h>
#include <eo>
#include <mo>

//! Variable Neighbors Search (VNS)
/*!
  Class which describes the algorithm for a Variable Neighbors Search.
*/

template < class EOT>
class moVNS : public moAlgo < EOT>
{
	//! Alias for the fitness.
	typedef typename EOT::Fitness Fitness;

public:

	//! Generic constructor
  /*!
    Generic constructor using a moExpl

    \param _explorer Vector of Neighborhoods.
    \param _full_evaluation The evaluation function.
  */

	moVNS(moExpl< EOT> & _explorer, eoEvalFunc < EOT> & _full_evaluation): explorer(_explorer), full_evaluation(_full_evaluation) {}


	//! Function which launches the VNS
  /*!
    The VNS has to improve a current solution.

    \param _solution a current solution to improve.
    \return true.
  */

	bool operator()(EOT & _solution) {
		bool change=false;
		int i = 0;

		EOT solution_initial=_solution;
		EOT solution_prime, solution_second;


		explorer.setIndice(i);

		while(i<explorer.size()) {
			solution_prime=solution_initial;
			if(solution_prime.invalid())
				full_evaluation(solution_prime);
			explorer(solution_prime, solution_second);
			if(solution_second.invalid())
				full_evaluation(solution_second);
			if(solution_second > solution_initial) {
				solution_initial=solution_second;
				change=true;
				if(i!= 0) {
					explorer.setIndice(0);
					i=0;
				}
			}
			else {
				i++;
				if(i<explorer.size())
					explorer.setIndice(i);
			}
		}
		_solution=solution_initial;
		return change;
	}

private:
	//Neighborhoods vector
	moExpl<EOT> & explorer;
	//The full evaluation function
	eoEvalFunc<EOT> & full_evaluation;
};

#endif
