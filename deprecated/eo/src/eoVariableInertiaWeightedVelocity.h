// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoVariableInertiaWeightedVelocity.h
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

#ifndef EOVARIABLEINERTIAWEIGHTEDVELOCITY_H
#define EOVARIABLEINERTIAWEIGHTEDVELOCITY_H

//-----------------------------------------------------------------------------
#include <eoVelocity.h>
#include <eoTopology.h>
#include <eoWeightUpdater.h>
#include <utils/eoRealVectorBounds.h>
#include <eoRealBoundModifier.h>
//-----------------------------------------------------------------------------



/** Inertia variable weight based velocity for particle swarm optimization. Derivated from abstract eoVelocity,
*   At step t: v(t+1)= w*v(t) + c1*r1* ( xbest(t)-x(t) ) + c2*r2* ( gbest(t) - x(t) )
*   w is updated each time the velocity performer is called.
*   (ci given and Ri chosen at random in [0;1]).
*
*   @ingroup Core
*   @ingroup Variators
*/
template < class POT > class eoVariableInertiaWeightedVelocity:public eoVelocity < POT >
{

public:

    /*
     * Each element for the velocity evaluation is expected to be of type VelocityType.
     */
    typedef typename POT::ParticleVelocityType VelocityType;

    /** Full constructor: Bounds and bound modifier required
    * @param _topology - The topology to get the global/local/other best
    * @param _weightUpdater - An eoWeightUpdater used to update the inertia weight
    * @param _c1 - The first learning factor used for the particle's best. Type must be POT::ParticleVelocityType
    * @param _c2 - The second learning factor used for the local/global best(s). Type must be POT::ParticleVelocityType
    * @param _bounds - An eoRealBaseVectorBounds: real bounds for real velocities.
    * If the velocities are not real, they won't be bounded by default. Should have a eoBounds ?
    * @param _bndsModifier - An eoRealBoundModifier used to modify the bounds (for real bounds only).
    * @param _gen - The eo random generator, default=rng
    */
    eoVariableInertiaWeightedVelocity (eoTopology < POT > & _topology,
                                       eoWeightUpdater<VelocityType> & _weightUpdater,
                                       const VelocityType & _c1,
                                       const VelocityType & _c2 ,
                                       eoRealVectorBounds & _bounds,
                                       eoRealBoundModifier & _bndsModifier,
                                       eoRng & _gen = rng):
            topology(_topology),
            weightUpdater(_weightUpdater),
            c1 (_c1),
            c2 (_c2),
            bounds(_bounds),
            bndsModifier(_bndsModifier),
            gen(_gen){}


    /** Constructor: No bound updater required <-> fixed bounds
       * @param _topology - The topology to get the global/local/other best
       * @param _weightUpdater - An eoWeightUpdater used to update the inertia weight
       * @param _c1 - The first learning factor used for the particle's best. Type must be POT::ParticleVelocityType
       * @param _c2 - The second learning factor used for the local/global best(s). Type must be POT::ParticleVelocityType
       * @param _bounds - An eoRealBaseVectorBounds: real bounds for real velocities.
       * If the velocities are not real, they won't be bounded by default. Should have a eoBounds ?
       * @param _gen - The eo random generator, default=rng
       */
    eoVariableInertiaWeightedVelocity (eoTopology < POT > & _topology,
                                       eoWeightUpdater<VelocityType> & _weightUpdater,
                                       const VelocityType & _c1,
                                       const VelocityType & _c2,
                                       eoRealVectorBounds & _bounds,
                                       eoRng & _gen = rng):
            topology(_topology),
            weightUpdater(_weightUpdater),
            c1 (_c1),
            c2 (_c2),
            bounds(_bounds),
            bndsModifier(dummyModifier),
            gen(_gen){}


    /** Constructor: Neither bounds nor bound updater required <-> free velocity
       * @param _topology - The topology to get the global/local/other best
       * @param _weightUpdater - An eoWeightUpdater used to update the inertia weight
       * @param _c1 - The first learning factor used for the particle's best. Type must be POT::ParticleVelocityType
       * @param _c2 - The second learning factor used for the local/global best(s). Type must be POT::ParticleVelocityType
       * @param _gen - The eo random generator, default=rng
       */
    eoVariableInertiaWeightedVelocity (eoTopology < POT > & _topology,
                                       eoWeightUpdater<VelocityType> & _weightUpdater,
                                       const VelocityType & _c1,
                                       const VelocityType & _c2,
                                       eoRng & _gen = rng):
            topology(_topology),
            weightUpdater(_weightUpdater),
            c1 (_c1),
            c2 (_c2),
            bounds(*(new eoRealVectorNoBounds(0))),
            bndsModifier(dummyModifier),
            gen(_gen)
    {}

    /**
     *  Evaluate the new velocities of the given particle. Need an indice to identify the particle
     * into the topology. Steps are :
     *  - evaluate r1 and r2, the customed learning factors
     *  - adjust the size of the bounds (even if dummy)
     *  - update the weight with the weightUpdater (use the dummy updater if there's no updater provided)
     *  - modify the bounds with the bounds modifier (use the dummy modifier if there's no modifier provided)
     * @param _po - A particle
     * @param _indice - The indice (into the topology) of the given particle
     */
    void operator  () (POT & _po,unsigned _indice)
    {
        VelocityType r1;
        VelocityType r2;

        VelocityType newVelocity;

        // cast the learning factors to VelocityType
        r1 = (VelocityType) rng.uniform (1) * c1;
        r2 = (VelocityType) rng.uniform (1) * c2;

        // need to resize the bounds even if there are dummy because of "isBounded" call
        bounds.adjust_size(_po.size());

        // update the inertia weight
        weightUpdater(weight);

        // assign the new velocities
        for (unsigned j = 0; j < _po.size (); j++)
        {
            newVelocity= weight * _po.velocities[j] + r1 * (_po.bestPositions[j] - _po[j]) +  r2 * (topology.best (_indice)[j] - _po[j]);

            /* modify the bounds */
            bndsModifier(bounds,j);

            /* check bounds */
            if (bounds.isMinBounded(j))
                newVelocity=(VelocityType)std::max(newVelocity,bounds.minimum(j));
            if (bounds.isMaxBounded(j))
                newVelocity=(VelocityType)std::min(newVelocity,bounds.maximum(j));

            _po.velocities[j]=newVelocity;
        }
    }

    /**
     * Update the neighborhood.
     */
    void updateNeighborhood(POT & _po,unsigned _indice)
    {
        topology.updateNeighborhood(_po,_indice);
    }



protected:
    eoTopology < POT > & topology;
    eoWeightUpdater<VelocityType> & weightUpdater;      // the updater used to make the weight evoluate
    const VelocityType & c1;    // learning factor 1
    const VelocityType  & c2;    // learning factor 2

        eoRealVectorBounds & bounds; // REAL bounds even if the velocity could be of another type.
    eoRealBoundModifier & bndsModifier;

    VelocityType weight;
    eoRng & gen;        // the random generator

    // If the bound modifier doesn't need to be used, use the dummy instance
    eoDummyRealBoundModifier dummyModifier;
};

#endif /*EOVARIABLEINERTIAWEIGHTEDVELOCITY_H*/
