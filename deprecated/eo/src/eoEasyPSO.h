// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoEasyPSO.h
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

#ifndef _EOEASYPSO_H
#define _EOEASYPSO_H

//-----------------------------------------------------------------------------
#include <eoContinue.h>
#include <eoPSO.h>
#include <eoVelocity.h>
#include <eoFlight.h>
//-----------------------------------------------------------------------------

/** An easy-to-use particle swarm algorithm.
* Use any particle, any flight, any topology...
*
*   The main steps are :
*         (The population  is expected to be already evaluated)
*        - for each generation and each particle pi
*        - evaluate the velocities
*               -- perform the fligth of pi
*       -- evaluate pi
*       -- update the neighborhoods
*
* @ingroup Algorithms
*/
template < class POT > class eoEasyPSO:public eoPSO < POT >
{
public:

    /** Full constructor
    * @param _init - An eoInitializerBase that initializes the topology, velocity, best particle(s)
    * @param _continuator - An eoContinue that manages the stopping criterion and the checkpointing system
    * @param _eval - An eoEvalFunc: the evaluation performer
    * @param _velocity - An eoVelocity that defines how to compute the velocities
    * @param _flight - An eoFlight that defines how to make the particle flying: that means how
    * to modify the positions according to the velocities
    */
    eoEasyPSO (
        eoInitializerBase <POT> &_init,
        eoContinue < POT > &_continuator,
        eoEvalFunc < POT > &_eval,
        eoVelocity < POT > &_velocity,
        eoFlight < POT > &_flight):
            init(_init),
            continuator (_continuator),
            eval (_eval),
            velocity (_velocity),
            flight (_flight)
    {}


    /** Constructor without eoFlight. For special cases when the flight is performed withing the velocity.
       * @param _init - An eoInitializerBase that initializes the topology, velocity, best particle(s)
       * @param _continuator - An eoContinue that manages the stopping criterion and the checkpointing system
       * @param _eval - An eoEvalFunc: the evaluation performer
       * @param _velocity - An eoVelocity that defines how to compute the velocities
    */
    eoEasyPSO (
        eoInitializerBase <POT> &_init,
        eoContinue < POT > &_continuator,
        eoEvalFunc < POT > &_eval,
        eoVelocity < POT > &_velocity):
            init(_init),
            continuator (_continuator),
            eval (_eval),
            velocity (_velocity),
            flight (dummyFlight)
    {}


        /* Constructor without eoInitializerBase. Assume the initialization is done before running the algorithm
    * @param _continuator - An eoContinue that manages the stopping criterion and the checkpointing system
    * @param _eval - An eoEvalFunc: the evaluation performer
    * @param _velocity - An eoVelocity that defines how to compute the velocities
    * @param _flight - An eoFlight that defines how to make the particle flying: that means how
    * to modify the positions according to the velocities
    */
    eoEasyPSO (
        eoContinue < POT > &_continuator,
        eoEvalFunc < POT > &_eval,
        eoVelocity < POT > &_velocity,
        eoFlight < POT > &_flight):
            init(dummyInit),
            continuator (_continuator),
            eval (_eval),
            velocity (_velocity),
            flight (_flight)
    {}


    /** Constructor without eoFlight nor eoInitializerBase. For special cases when the flight is performed withing the velocity.
       * @param _continuator - An eoContinue that manages the stopping criterion and the checkpointing system
       * @param _eval - An eoEvalFunc: the evaluation performer
       * @param _velocity - An eoVelocity that defines how to compute the velocities
    */
    eoEasyPSO (
        eoContinue < POT > &_continuator,
        eoEvalFunc < POT > &_eval,
        eoVelocity < POT > &_velocity):
            init(dummyInit),
            continuator (_continuator),
            eval (_eval),
            velocity (_velocity),
            flight (dummyFlight)
    {}

    /// Apply a few iteration of flight to the population (=swarm).
    virtual void operator  () (eoPop < POT > &_pop)
    {
        try
        {
            // initializes the topology, velocity, best particle(s)
            init();
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

                    // update the topology (particle and local/global best(s))
                    velocity.updateNeighborhood(_pop[idx],idx);
                }

            }
            while (continuator (_pop));

        }
        catch (std::exception & e)
        {
            std::string s = e.what ();
            s.append (" in eoEasyPSO");
            throw std::runtime_error (s);
        }

    }

protected:
    eoInitializerBase <POT> &init;
    eoContinue < POT > &continuator;
    eoEvalFunc < POT > &eval;
    eoVelocity < POT > &velocity;
    eoFlight < POT > &flight;

         // if the flight does not need to be used, use the dummy flight instance
        class eoDummyFlight:public eoFlight < POT >
        {
         public:
        eoDummyFlight () {}
        void operator  () (POT & _po) {}
        }dummyFlight;

        // if the initializer does not need to be used, use the dummy one instead
        class eoDummyInitializer:public eoInitializerBase < POT >
        {
         public:
        eoDummyInitializer () {}
        void operator  () (POT & _po) {}
        }dummyInit;

};
/**
 * @example t-eoEasyPSO.cpp
 * Example of a test program building a PSO algorithm.
 */

#endif /*_EOEASYPSO_H*/
