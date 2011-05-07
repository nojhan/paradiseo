// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoConstrictedVelocity.h
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
             clive.canape@inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef EOCONSTRICTEDVELOCITY_H
#define EOCONSTRICTEDVELOCITY_H

//-----------------------------------------------------------------------------
#include <eoFunctor.h>
#include <utils/eoRNG.h>
#include <eoPop.h>
#include <utils/eoRealVectorBounds.h>
#include <eoTopology.h>
//-----------------------------------------------------------------------------


/** Constricted velocity performer for particle swarm optimization. Derivated from abstract eoVelocity,
*   At step t+1 : v(t+1)= C * [ v(t) + c1*r1 * (xbest(t)-x(t)) + c2*r2 * (gbest(t) - x(t)) ]
*   C is fixed for all the particles and all the generations.
*   Default C = 2 * k / abs(2 - P - sqrt (P*(P-4)))
*   (ci and C given;P=c1*r1 + c2*r2 ; Ri chosen at random *   in [0;1])
*
*   @ingroup Variators
*/
template < class POT > class eoConstrictedVelocity:public eoVelocity < POT >
{

public:

    /*
    * Each element for the velocity evaluation is expected to be of type VelocityType.
    */
    typedef typename POT::ParticleVelocityType VelocityType;

    /** Full constructor: Bounds and bound modifier required
    * @param _topology - The topology to get the global/local/other best
    * @param _coeff - The constriction coefficient
    * @param _c1 - The first learning factor used for the particle's best. Type must be POT::ParticleVelocityType
    * @param _c2 - The second learning factor used for the local/global best(s). Type must be POT::ParticleVelocityType
    * @param _bounds - An eoRealBaseVectorBounds: real bounds for real velocities.
    * If the velocities are not real, they won't be bounded by default. Should have a eoBounds ?
    * @param _bndsModifier - An eoRealBoundModifier used to modify the bounds (for real bounds only).
    * @param _gen - The eo random generator, default=rng
    */
    eoConstrictedVelocity (eoTopology < POT > & _topology,
                           const VelocityType & _coeff,
                           const VelocityType & _c1,
                           const VelocityType & _c2 ,
                           eoRealVectorBounds & _bounds,
                           eoRealBoundModifier & _bndsModifier,
                           eoRng & _gen = rng):
            topology(_topology),
            coeff(_coeff),
            c1 (_c1),
            c2 (_c2),
            bounds(_bounds),
            bndsModifier(_bndsModifier),
            gen(_gen){}


    /** Constructor: No bound updater required <-> fixed bounds
       * @param _topology - The topology to get the global/local/other best
       * @param _coeff - The constriction coefficient
       * @param _c1 - The first learning factor used for the particle's best. Type must be POT::ParticleVelocityType
       * @param _c2 - The second learning factor used for the local/global best(s). Type must be POT::ParticleVelocityType
       * @param _bounds - An eoRealBaseVectorBounds: real bounds for real velocities.
       * If the velocities are not real, they won't be bounded by default. Should have a eoBounds ?
       * @param _gen - The eo random generator, default=rng
       */
    eoConstrictedVelocity (eoTopology < POT > & _topology,
                           const VelocityType & _coeff,
                           const VelocityType & _c1,
                           const VelocityType & _c2,
                           eoRealVectorBounds & _bounds,
                           eoRng & _gen = rng):
            topology(_topology),
            coeff(_coeff),
            c1 (_c1),
            c2 (_c2),
            bounds(_bounds),
            bndsModifier(dummyModifier),
            gen(_gen){}


    /** Constructor: Neither bounds nor bound updater required <-> free velocity
       * @param _topology - The topology to get the global/local/other best
       * @param _coeff - The constriction coefficient
       * @param _c1 - The first learning factor used for the particle's best. Type must be POT::ParticleVelocityType
       * @param _c2 - The second learning factor used for the local/global best(s). Type must be POT::ParticleVelocityType
       * @param _gen - The eo random generator, default=rng
       */
    eoConstrictedVelocity (eoTopology < POT > & _topology,
                           const VelocityType & _coeff,
                           const VelocityType & _c1,
                           const VelocityType & _c2,
                           eoRng & _gen = rng):
            topology(_topology),
            coeff(_coeff),
            c1 (_c1),
            c2 (_c2),
            bounds(*(new eoRealVectorNoBounds(0))),
            bndsModifier(dummyModifier),
            gen(_gen){}


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
            newVelocity= coeff *  (_po.velocities[j] + r1 * (_po.bestPositions[j] - _po[j]) +  r2 * (topology.best (_indice)[j] - _po[j]));

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
     * Update the neighborhood of a particle.
     */
    void updateNeighborhood(POT & _po,unsigned _indice)
    {
        topology.updateNeighborhood(_po,_indice);
    }

        //! eoTopology<POT> getTopology
    //! @return topology

        eoTopology<POT> & getTopology ()
        {
                return topology;
        }

protected:
    eoTopology < POT > & topology;
    const VelocityType & c1;    // learning factor 1
    const VelocityType  & c2;    // learning factor 2
    const VelocityType & coeff;   // the fixed constriction coefficient
    eoRng & gen;        // the random generator

    eoRealVectorBounds & bounds; // REAL bounds even if the velocity could be of another type.
    eoRealBoundModifier & bndsModifier;

    // If the bound modifier doesn't need to be used, use the dummy instance
    eoDummyRealBoundModifier dummyModifier;
};


#endif /*EOCONSTRICTEDVELOCITY_H */
