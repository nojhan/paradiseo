// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoFixedInertiaWeightedVelocity.h
// (c) OPAC 2007
/*
    Contact: paradiseo-help@lists.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef EOFIXEDINERTIAWEIGHTEDVELOCITY_H
#define EOFIXEDINERTIAWEIGHTEDVELOCITY_H

//-----------------------------------------------------------------------------
#include <eoFunctor.h>
#include <utils/eoRNG.h>
#include <eoPop.h>
#include <utils/eoRealVectorBounds.h>
#include <eoTopology.h>
//-----------------------------------------------------------------------------


/** Inertia weight based velocity performer. Derivated from abstract eoVelocity,
*   At step t+1 : v(t+1)= w * v(t) + delta1 * ( xbest(t)-x(t) ) + delta2 * ( gbest(t) - x(t) )
*   with delta1= c1 * R1 and delta2= c2 * R2 (ci and w given; Ri chosen at random
*   in [0;1])
*/
template < class POT > class eoFixedInertiaWeightedVelocity:public eoVelocity < POT >
{

public:

    /*
    * Each element for the velocity evaluation is expected to be of type VelocityType.
    */
    typedef typename POT::ParticleVelocityType VelocityType;

    /** Full constructor: Bounds and bound modifier required
    * @param _topology - The topology to get the global/local/other best
    * @param _c1 - The first learning factor used for the particle's best. Type must be POT::ParticleVelocityType 
    * @param _c2 - The second learning factor used for the local/global best(s). Type must be POT::ParticleVelocityType 
    * @param _bounds - An eoRealBaseVectorBounds: real bounds for real velocities. 
    * If the velocities are not real, they won't be bounded by default. Should have a eoBounds ?
    * @param _boundsModifier - An eoRealBoundModifier used to modify the bounds (for real bounds only).
    * @param _gen - The eo random generator, default=rng
    */
    eoFixedInertiaWeightedVelocity (eoTopology < POT > & _topology,
                                    const VelocityType _c1,
                                    const VelocityType _c2 ,
                                    eoRealVectorBounds & _bounds,
                                    eoRealBoundModifier & _bndsModifier,
                                    eoRng & _gen = rng):
            topology(_topology),
            c1 (_c1),
            c2 (_c2),
            bounds(_bounds),
            bndsModifier(_bndsModifier),
            gen(_gen){}


    /** Constructor: No bound updater required <-> fixed bounds
       * @param _topology - The topology to get the global/local/other best
       * @param _c1 - The first learning factor used for the particle's best. Type must be POT::ParticleVelocityType 
       * @param _c2 - The second learning factor used for the local/global best(s). Type must be POT::ParticleVelocityType 
       * @param _bounds - An eoRealBaseVectorBounds: real bounds for real velocities. 
       * If the velocities are not real, they won't be bounded by default. Should have a eoBounds ?
       * @param _gen - The eo random generator, default=rng
       */
    eoFixedInertiaWeightedVelocity (eoTopology < POT > & _topology,
                                    const VelocityType _c1,
                                    const VelocityType _c2,
                                    eoRealVectorBounds & _bounds,
                                    eoRng & _gen = rng):
            topology(_topology),
            c1 (_c1),
            c2 (_c2),
            bounds(_bounds),
            bndsModifier(dummyModifier),
            gen(_gen){}


    /** Constructor: Neither bounds nor bound updater required <-> free velocity
       * @param _topology - The topology to get the global/local/other best
       * @param _c1 - The first learning factor used for the particle's best. Type must be POT::ParticleVelocityType 
       * @param _c2 - The second learning factor used for the local/global best(s). Type must be POT::ParticleVelocityType 
       * @param _gen - The eo random generator, default=rng
       */
    eoFixedInertiaWeightedVelocity (eoTopology < POT > & _topology,
                                    const VelocityType _c1,
                                    const VelocityType _c2,
                                    eoRng & _gen = rng):
            topology(_topology),
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

            _po.velocities[j]=(VelocityType)newVelocity;
        }
    }

    /**
     * Update the topology.
     */
    void updateTopology(POT & _po,unsigned _indice)
    {
        topology.update(_po,_indice);
    }



protected:
    eoTopology < POT > & topology;
    const VelocityType c1;  	// learning factor 1
    const VelocityType  c2; 	 // learning factor 2
    VelocityType weight;   // the fixed weight
    eoRng & gen; 	// the random generator

    eoRealVectorBounds & bounds; // REAL bounds even if the velocity could be of another type.
    eoRealBoundModifier & bndsModifier;

    // If the bound modifier doesn't need to be used, use the dummy instance
    eoDummyRealBoundModifier dummyModifier;
};


#endif /*EOFIXEDINERTIAWEIGHTEDVELOCITY_H */

