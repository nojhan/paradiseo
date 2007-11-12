// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoInitializer.h
// (c) OPAC Team, INRIA, 2007
/*
    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Contact: clive.canape@inria.fr

             
 */
//-----------------------------------------------------------------------------

#ifndef _eoInitializer_H
#define _eoInitializer_H

#include <utils/eoRealVectorBounds.h>
#include <eoVelocityInit.h>
#include <eoPop.h>
#include <eoParticleBestInit.h>


/*
 * Abstract class for initialization of algorithm PSO
 */
template <class POT> class eoInitializerBase : public eoFunctorBase
{
	public :

		virtual ~eoInitializerBase() {}

		virtual void operator()(){};
};

/**
	Base (name) class for Initialization of algorithm PSO

	@see eoInitializerBase eoUF apply
*/
template <class POT> class eoInitializer : public eoInitializerBase <POT>
{
	public:

	//!	Constructor
	//! @param _proc Evaluation function
	//! @param _initVelo Initialization of the velocity
	//! @param _initBest Initialization of the best
	//! @param _pop Population 
	eoInitializer(
					eoUF<POT&, void>& _proc,
					eoVelocityInit < POT > &_initVelo, 
					eoParticleBestInit <POT> &_initBest,
					eoPop < POT > &_pop
				 ) : proc(_proc), initVelo(_initVelo), initBest(_initBest)
	{
		apply(proc, _pop);
        apply < POT > (initVelo, _pop);
    	apply < POT > (initBest, _pop);
	}
	
	//! Give the name of the class
	//! @return The name of the class
	virtual std::string className (void) const
    {
        return "eoInitializer";
    }
	
	private :
	
	/*
		@param proc First evaluation
		@param initVelo Initialization of the velocity
		@param initBest Initialization of the best
		
	*/
	eoUF<POT&, void>& proc;
	eoVelocityInit < POT > & initVelo;
	eoParticleBestInit <POT> & initBest;
};
#endif

	
	
