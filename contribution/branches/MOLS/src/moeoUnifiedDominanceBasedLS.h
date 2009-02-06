/*
* <moeoUnifiedDominanceBasedLS.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2008
* (C) OPAC Team, LIFL, 2002-2008
*
* Arnaud Liefooghe
* Jérémie Humeau
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
//-----------------------------------------------------------------------------

#ifndef _MOEOUNIFIEDDOMINANCEBASEDLS_H
#define _MOEOUNIFIEDDOMINANCEBASEDLS_H

#include <eo>
#include <moeo>
#include <moeoPopNeighborhoodExplorer.h>
#include <moeoPopLS.h>

/**
 * An easy class to design multi-objective evolutionary algorithms.
 */
template < class Move >
class moeoUnifiedDominanceBasedLS : public moeoPopLS < Move >
{
  //! Alias for the type
  typedef typename Move::EOType MOEOT;

public:

	moeoUnifiedDominanceBasedLS(
			eoContinue < MOEOT > & _continuator,
			//moeoContinue < MOEOT > & _naturalContinuator,
			eoEvalFunc < MOEOT > & _full_evaluation,
			eoPopEvalFunc < MOEOT > & _popEval,
			moeoArchive < MOEOT > & _archive,
			moeoPopNeighborhoodExplorer < Move > & _explorer
	):continuator(_continuator), naturalContinuator(defaultContinuator), full_evaluation(_full_evaluation), popEval(_popEval), archive(_archive), explorer(_explorer)
	{}

	moeoUnifiedDominanceBasedLS(
			eoContinue < MOEOT > & _continuator,
			moeoPopNeighborhoodExplorer < Move > & _explorer
	):continuator(_continuator), naturalContinuator(defaultContinuator), full_evaluation(dummyEval),loopEval(dummyEval), popEval(loopEval), archive(defaultArchive), explorer(_explorer)
	{}

    /**
     * Applies a few generation of evolution to the population _pop.
     * @param _pop the population
     */
    virtual void operator()(eoPop < MOEOT > & _pop)
    {
    	eoPop < MOEOT > tmp_pop;
		popEval(tmp_pop, _pop);// A first eval of pop.

    	archive(_pop);

    	do{
    		tmp_pop.resize(0);
        	//"perturber" la population
    		explorer(archive, tmp_pop);
        	//mise à jour de la pop ou archive
    		archive(tmp_pop);
    	}
    	while(continuator(tmp_pop) && naturalContinuator(archive));
    }

protected:

	eoContinue < MOEOT > & continuator;

	template<class MOEOT>
	class moeoContinue : public eoUF < eoPop < MOEOT > &, bool >
	{
	public:

		moeoContinue(){}

		virtual bool operator()(eoPop < MOEOT > & _pop){
			bool res = false;
			unsigned int i=0;
			while(!res && i < _pop.size()){
				res = (_pop[i].flag() == 0);
				i++;
			}
			return res;
		}
	};

	moeoContinue < MOEOT > defaultContinuator;

	moeoContinue < MOEOT > & naturalContinuator;

    /** a dummy eval */
    class eoDummyEval : public eoEvalFunc < MOEOT >
    {
    public:
        void operator()(MOEOT &) {}
    }
    dummyEval;

	eoEvalFunc < MOEOT > & full_evaluation;
	eoPopLoopEval < MOEOT > loopEval;
	eoPopEvalFunc < MOEOT > & popEval;

	moeoUnboundedArchive < MOEOT > defaultArchive;
	moeoArchive < MOEOT > & archive;

	moeoPopNeighborhoodExplorer < Move > & explorer;

};

#endif /*MOEOUNIFIEDDOMINANCEBASEDLS_H_*/
