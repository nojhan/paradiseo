// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoSyncEasyPSO.h
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

#ifndef _EOSYNCEASYPSO_H
#define _EOSYNCEASYPSO_H

//-----------------------------------------------------------------------------
#include <eoContinue.h>
#include <eoPopEvalFunc.h>
#include <eoPSO.h>
#include <eoVelocity.h>
#include <eoFlight.h>
#include <eoDummyFlight.h>
//-----------------------------------------------------------------------------

/** An easy-to-use synchronous particle swarm algorithm; you can use any particle,
*   any flight, any topology... The main steps are :
* 	 - perform a first evaluation of the population
* 	 - for each generation
* 	 - evaluate ALL the velocities
* 	 	-- perform the fligth of ALL the particles
*    	-- evaluate ALL the particles
*    	-- update the neighborhoods
*/
template < class POT > class eoSyncEasyPSO:public eoPSO < POT >
{
public:

    /** Full constructor
    * @param _continuator - An eoContinue that manages the stopping criterion and the checkpointing system
    * @param _eval - An eoEvalFunc: the evaluation performer
    * @param _velocity - An eoVelocity that defines how to compute the velocities
    * @param _flight - An eoFlight that defines how to make the particle flying: that means how 
    * to modify the positions according to the velocities
    */
    eoSyncEasyPSO (
        eoContinue < POT > &_continuator,
        eoEvalFunc < POT > &_eval,
        eoVelocity < POT > &_velocity,
        eoFlight < POT > &_flight):
            continuator (_continuator),
            eval (_eval),
            loopEval(_eval),
            popEval(loopEval),
            velocity (_velocity),
            flight (_flight){}


    /** Constructor without eoFlight. For special cases when the flight is performed withing the velocity.
       * @param _continuator - An eoContinue that manages the stopping criterion and the checkpointing system
       * @param _eval - An eoEvalFunc: the evaluation performer
       * @param _velocity - An eoVelocity that defines how to compute the velocities
    */
    eoSyncEasyPSO (
        eoContinue < POT > &_continuator,
        eoEvalFunc < POT > &_eval,
        eoVelocity < POT > &_velocity):
            continuator (_continuator),
            eval (_eval),
            loopEval(_eval),
            popEval(loopEval),
            velocity (_velocity),
            flight (dummyFlight){}



    /// Apply a few iteration of flight to the population (=swarm).
    virtual void operator  () (eoPop < POT > &_pop)
    {
        try
        {
            // just to use a loop eval
            eoPop<POT> empty_pop;

            do
            {
                // perform velocity evaluation
                velocity.apply (_pop);

                // apply the flight
                flight.apply (_pop);

                // evaluate the position (with a loop eval, empty_swarm IS USELESS)
                loopEval(empty_pop,_pop);

                // update the topology (particle and local/global best(s))
                velocity.updateNeighborhood(_pop);


            }while (continuator (_pop));

        }
        catch (std::exception & e)
        {
            std::string s = e.what ();
            s.append (" in eoSyncEasyPSO");
            throw std::runtime_error (s);
        }

    }

private:
    eoContinue < POT > &continuator;

    eoEvalFunc < POT > &eval;
    eoPopLoopEval<POT>        loopEval;
    eoPopEvalFunc<POT>&       popEval;

    eoVelocity < POT > &velocity;
    eoFlight < POT > &flight;

    // if the flight does not need to be used, use the dummy flight instance
    eoDummyFlight<POT> dummyFlight;
};


#endif /*_EOSYNCEASYPSO_H*/
