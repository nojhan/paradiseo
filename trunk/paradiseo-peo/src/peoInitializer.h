/* 
* <peoInitializer.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, INRIA, 2007
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

#ifndef _peoInitializer_H
#define _peoInitializer_H

/**
	Base (name) class for parallel initialization of algorithm PSO

	@see eoInitializerBase
*/

template <class POT> class peoInitializer : public eoInitializerBase <POT>
{
	public:
	

	//!	Constructor
	//! @param _proc Evaluation function
	//! @param _initVelo Initialization of the velocity
	//! @param _initBest Initialization of the best
	//! @param _pop Population 
	peoInitializer(
					peoPopEval< POT >& _proc,
					eoVelocityInit < POT > &_initVelo, 
					eoParticleBestInit <POT> &_initBest,
					eoPop < POT > &_pop
				 ) : proc(_proc), initVelo(_initVelo), initBest(_initBest)
	{
		pop = &_pop;
	}
	
	//! Give the name of the class
	//! @return The name of the class
	virtual std::string className (void) const
    {
        return "peoInitializer";
    }
    
    //! void operator ()
	//! Parallel initialization of the population
	virtual void operator()()
	{
		proc(*pop);
		apply < POT > (initVelo, *pop);
    	apply < POT > (initBest, *pop);
	}
	
	private :
	
	/*
		@param proc First evaluation
		@param initVelo Initialization of the velocity
		@param initBest Initialization of the best
		@param pop Population		
	*/
	peoPopEval< POT >& proc;
	eoVelocityInit < POT > & initVelo;
	eoParticleBestInit <POT> & initBest;
	eoPop <POT> * pop;
};
#endif

	
	
