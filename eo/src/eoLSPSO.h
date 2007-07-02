// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoLSPSO.h
// (c) OPAC 2007
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

    Contact: thomas.legrand@lifl.fr
 */
//-----------------------------------------------------------------------------

#ifndef _EOLSPSO_H
#define _EOLSPSO_H

//-----------------------------------------------------------------------------
#include <eoPSO.h>
#include <eoContinue.h>
#include <eoStandardFlight.h>
#include <eoLinearTopology.h>
#include <eoStandardVelocity.h>
#include <utils/eoRealVectorBounds.h>
#include <eoRealBoundModifier.h>
//-----------------------------------------------------------------------------

/**
 * A Linear ("L"inear topology) Standard ("S"tandard velocity) PSO.
 * You can use it with or without bounds on the velocity.
 * No bound for the flight (no bounds for the positions).
 *
 */
template < class POT > class eoLSPSO:public eoPSO < POT >
{
public:


    typedef typename POT::ParticleVelocityType VelocityType;

    /** Full constructor
    * @param _continuator - An eoContinue that manages the stopping criterion and the checkpointing system
    * @param _eval - An eoEvalFunc: the evaluation performer
    * @param _c1 - The first learning factor used for the particle's best. Type must be POT::ParticleVelocityType 
    * @param _c2 - The second learning factor used for the local/global best(s). Type must be POT::ParticleVelocityType 
    * @param _neighborhoodSize - The size of each neighborhood of the linear topology
    * @param _bounds - An eoRealBaseVectorBounds: real bounds for real velocities. 
    * If the velocities are not real, they won't be bounded by default. Should have a eoBounds ?
    * @param _boundsModifier - An eoRealBoundModifier used to modify the bounds (for real bounds only)
    */
    eoLSPSO (
        eoContinue < POT > &_continuator,
        eoEvalFunc < POT > &_eval,
        const VelocityType & _c1,
        const VelocityType & _c2 ,
        const unsigned _neighborhoodSize,
        eoRealVectorBounds & _bounds,
        eoRealBoundModifier & _bndsModifier):
            continuator (_continuator),
            eval (_eval),
            topology(eoLinearTopology<POT>(_neighborhoodSize)),
            velocity(eoStandardVelocity<POT>(topology,_c1,_c2,_bounds,_bndsModifier)),
            neighborhoodSize(_neighborhoodSize),
            bounds(_bounds),
            boundsModifier(_bndsModifier)
    {}

    /** Constructor without bound modifier.
    * @param _continuator - An eoContinue that manages the stopping criterion and the checkpointing system
    * @param _eval - An eoEvalFunc: the evaluation performer
    * @param _c1 - The first learning factor used for the particle's best. Type must be POT::ParticleVelocityType 
    * @param _c2 - The second learning factor used for the local/global best(s). Type must be POT::ParticleVelocityType 
    * @param _neighborhoodSize - The size of each neighborhood of the linear topology
    * @param _bounds - An eoRealBaseVectorBounds: real bounds for real velocities. 
    * If the velocities are not real, they won't be bounded by default. Should have a eoBounds ?
    */
    eoLSPSO (
        eoContinue < POT > &_continuator,
        eoEvalFunc < POT > &_eval,
        const VelocityType & _c1,
        const VelocityType & _c2 ,
        const unsigned _neighborhoodSize,
        eoRealVectorBounds & _bounds):
            continuator (_continuator),
            eval (_eval),
            topology(eoLinearTopology<POT>(_neighborhoodSize)),
            velocity(eoStandardVelocity<POT>(topology,_c1,_c2,_bounds)),
            neighborhoodSize(_neighborhoodSize),
            bounds(_bounds),
            boundsModifier(dummyModifier)
    {}


    /** Constructor without bounds nor bound modifier.
    * @param _continuator - An eoContinue that manages the stopping criterion and the checkpointing system
    * @param _eval - An eoEvalFunc: the evaluation performer
    * @param _c1 - The first learning factor used for the particle's best. Type must be POT::ParticleVelocityType 
    * @param _c2 - The second learning factor used for the local/global best(s). Type must be POT::ParticleVelocityType  
    * @param _neighborhoodSize - The size of each neighborhood of the linear topology
    * If the velocities are not real, they won't be bounded by default. Should have a eoBounds ?
    */
    eoLSPSO (
        eoContinue < POT > &_continuator,
        eoEvalFunc < POT > &_eval,
        const VelocityType & _c1,
        const VelocityType & _c2,
        const unsigned _neighborhoodSize):
            continuator (_continuator),
            eval (_eval),
            topology(eoLinearTopology<POT>(_neighborhoodSize)),
            velocity(eoStandardVelocity<POT>(topology,_c1,_c2)),
            neighborhoodSize(_neighborhoodSize),
            bounds(*(new eoRealVectorNoBounds(0))),
            boundsModifier(dummyModifier)
    {}


    /// Apply a few iteration of flight to the population (=swarm).
    virtual void operator  () (eoPop < POT > &_pop)
    {
        try
        {
            // setup the topology (done once)
            topology.setup(_pop);

            do
            {
                // loop over all the particles for the current iteration
                for (unsigned idx = 0; idx < _pop.size (); idx++)
                {
                    // perform velocity evaluation
                    velocity (_pop[idx],idx);

                    // apply the flight
                    flight (_pop[idx]);

                    // evaluate the position
                    eval (_pop[idx]);

                    // update the topology (particle and the global bests)
                    velocity.updateNeighborhood(_pop[idx],idx);
                }

            } while (continuator (_pop));

        }catch (std::exception & e)
        {
            std::string s = e.what ();
            s.append (" in eoLSPSO");
            throw std::runtime_error (s);
        }

    }

protected:
    eoContinue < POT > &continuator;
    eoEvalFunc < POT > &eval;

    eoStandardVelocity < POT > velocity;
    eoStandardFlight < POT > flight;

    const unsigned neighborhoodSize;
    eoLinearTopology<POT> topology;

    eoRealVectorBounds bounds; // REAL bounds even if the velocity could be of another type.
    eoRealBoundModifier & boundsModifier;

    // If the bound modifier doesn't need to be used, use the dummy instance
    eoDummyRealBoundModifier dummyModifier;

};


#endif /*_EOLSPSO_H*/
