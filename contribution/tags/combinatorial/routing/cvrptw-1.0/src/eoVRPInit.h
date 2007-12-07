/*
 * Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
 * (C) OPAC Team, LIFL, 2002-2007
 *
 * (c) Antonio LaTorre <atorre@fi.upm.es>, 2007
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

#ifndef _eoVRPInit_h
#define _eoVRPInit_h

// The base definition of eoInit
#include <eoInit.h>

// Utilities for the VRP-TW problem
#include "eoVRPUtils.h"

/**
  * \def ALFA
  * Constant used by "selectCheapestClient" method.
  * \def BETA
  * Constant used by "selectCheapestClient" method.
  * \def GAMMA
  * Constant used by "selectCheapestClient" method.
  */

#define ALFA                 0.7
#define BETA                 0.1
#define GAMMA                0.2

/**
  * \class eoVRPInit eoVRPInit.h
  * \brief Class defining the initializer functor.
  * This class initializes an individual of the VRP problem using
  * an heuristic initializer.
  */

class eoVRPInit: public eoInit <eoVRP> {

public:

    /**
      * \brief Default constructor: nothing to do here.
      */

    eoVRPInit () {

        unsigned sz = eoVRPUtils::clients.size ();

        if (sz <= 0) {

            std::cerr << "Error: the dataset MUST be read before creating the initializer object." << std::endl;
            abort ();

        }

        mSeedsUsedCount = 0;

        for (unsigned i = 0; i < sz; i++)
            mSeedsUsed.push_back (false);

    }


    /**
      * \brief Functor member.
      * Initializes a genotype using an heuristic initializer.
      * \param _gen Generally a genotype that has been default-constructed.
      *             Whatever it contains will be lost.
      */

    void operator () (eoVRP& _gen) {

        HeuristicInitialization (_gen);

    }


private:

    unsigned mSeedsUsedCount;        /**< Number of clients already used as seeds. */
    std::vector<bool> mSeedsUsed;    /**< Vector storing if a client has been used as a seed or not.  */


    /**
      * \brief Heuristic initializer.
      * This initializer constructs and individual from routes. Each route is built
      * in a constructive way. The first client of each route is selected trying to
      * maximize a function depending on its time window and how far it is from the depot.
      * We always try to select the hardest clients as seeds. Used seeds are stored
      * so that different seeds are selected for different individuals and thus guarantee
      * the diversity of the initial population.
      * \param _gen The individual to be initialized.
      */

    void HeuristicInitialization (eoVRP& _gen) {

        // Aux var to override seed used checking
        bool seedCheckingOverride = false;

        // Erase any previous contents
        _gen.clear ();

        // Aux vector to store unvisited clients
        std::vector<int> unvisited;

        // And an index to point to the last unvisited cutomer
        int unvisitedIdx = eoVRPUtils::clients.size () - 2;

        // Initialization of the unvisited vector with all the clients
        for (unsigned i = 1; i < eoVRPUtils::clients.size (); i++)
            unvisited.push_back (i);

        // Main loop: keep iterating until all clients are visited
        while (unvisitedIdx >= 0) {

            // Aux var to store the new route
            Route route;

            createNewRoute (unvisited, unvisitedIdx, seedCheckingOverride, route);
            seedCheckingOverride = true;

            for (unsigned i = 0; i < route.size (); i++)
                _gen.push_back (route [i]);

        }

        // Invalidates the genotype forcing its re-evaluation
        _gen.invalidate ();

    }


    /**
      * \brief Creates a new route.
      * Creates a new route selecting the best (hardest) client as seed and then adding
      * the cheapest clients until one of the constraints (time window or vehicle's capacity)
      * is broken.
      * \param _unvisited Vector of unvisited and thus available clients for constructing the new route.
      * \param _unvisitedIdx Position of the last univisted client in _unvisited vector.
      * \param _seedCheckingOverride If true, it overrides the seed checking mecanism. It must be
      *         always false for the first route and then true for the following ones.
      *         This way we will preserve diversity in our initial population as every individual
      *         will be initialized from a different initial route.
      * \param _route The brand new route we have constructed.
      * \return True if everything went ok.
      */

    bool createNewRoute (std::vector<int>& _unvisited, int& _unvisitedIdx,
                         bool _seedCheckingOverride, Route& _route         ) {

        // Selection of seed client for current route
        unsigned seed = selectBestClientAsSeed (_unvisited, _unvisitedIdx, _seedCheckingOverride);

        // Add the seed client to the route
        _route.push_back (_unvisited [seed]);

        // Mark the client as already selected as a main seed
        // (i.e., as a seed for the first subroute of an individual)
        if (!_seedCheckingOverride) {

            mSeedsUsed [_unvisited [seed]] = true;
            mSeedsUsedCount++;

            if (mSeedsUsedCount == mSeedsUsed.size () - 1) {

                mSeedsUsedCount = 0;

                for (unsigned i = 0; i < mSeedsUsed.size (); i++)
                    mSeedsUsed [i] = false;

            }

        }

        // Delete the selected client from the unvisited vector
        _unvisited [seed] = _unvisited [_unvisitedIdx];
        _unvisitedIdx--;

        bool feasibleInsert = true;

        while (feasibleInsert && _unvisitedIdx >= 0) {

            // Aux var to store the position to insert new clients in the route
            Route::iterator it;

            unsigned next;

            if (selectBestInsertion (_unvisited, _unvisitedIdx, _route, next, it)) {

                if (it == _route.end ())
                    _route.insert (_route.begin (), _unvisited [next]);
                else
                    _route.insert (it + 1, _unvisited [next]);

                _unvisited [next] = _unvisited [_unvisitedIdx];
                _unvisitedIdx--;

            }
            else
                feasibleInsert = false;

        }

        return true;

    }


    /**
      * \brief Selects the best client and the best position for its insertion in a given route.
      * Given a subroute, this method tries to find the best client and the best position for it
      * among all the univisited clients.
      * \param _unvisited Vector of unvisited and thus available clients for constructing the new route.
      * \param _unvisitedIdx Position of the last univisted client in _unvisited vector.
      * \param _route The route where we are trying to insert a new client.
      * \param _nextClient A return value. The selected client to be inserted.
      * \param _it A return value. The position for selected client to be inserted.
      * \return True if a new insertion is possible. False otherwise.
      */

    bool selectBestInsertion (std::vector<int>& _unvisited, unsigned _unvisitedIdx, Route& _route,
                              unsigned& _nextClient, Route::iterator& _it                          ) {

        double minCost = 999999999;
        bool found = false;

        bool insertionPossible = false;
        double cost = 0.0;

        for (unsigned i = 0; i < _unvisitedIdx; i++) {

            insertionPossible = evaluateInsertion (_route, _unvisited [i], -1, cost);

            if (insertionPossible && cost < minCost) {

                _it = _route.end ();
                _nextClient = i;
                minCost = cost;
                found = true;

            }

        }

        for (Route::iterator it = _route.begin (); it != _route.end (); it++) {

            for (unsigned i = 0; i < _unvisitedIdx; i++) {

                insertionPossible = evaluateInsertion (_route, _unvisited [i], *it, cost);

                if (insertionPossible && cost < minCost) {

                    _it = it;
                    _nextClient = i;
                    minCost = cost;
                    found = true;

                }

            }

        }

        return found;

    }


    /**
      * \brief Evaluates the feasibility and the cost of inserting a new client in a given subroute.
      * Given a subroute, this method tries evaluates if it is possible to insert a client in a position.
      * It will return the cost of the resulting route if this insertion is possible.
      * \param _route The route where we are trying to insert a new client.
      * \param _newClient The client we are trying to insert.
      * \param _afterClient The position of insertion.
      * \param _cost A return value. The cost of inserting the given client at the given position.
      * \return True if the new insertion is possible. False otherwise.
      */

    bool evaluateInsertion (Route& _route, unsigned _newClient, int _afterClient, double& _cost) {

        // First of all, we check for capacity constraint to be satisfied
        // before trying to insert the new client in the route
        double demand = 0.0;

        // Cummulate the demand of all the clients in the current route
        for (unsigned i = 0; i < _route.size (); i++)
            demand += eoVRPUtils::clients [i].demand;

        // And then the demand of the new client
        demand += eoVRPUtils::clients [_newClient].demand;

        // And finally, check if the cummulated demand exceeds vehicle's capacity
        if (demand > VEHICLE_CAPACITY)
            return false;

        // Now check for insertion position and TW constraints
        double readyTime, dueTime, serviceTime;

        // If the client must be inserted after client "-1" that means that it
        // has to be inserted at the very beginning of the route
        if (_afterClient == - 1) {

            // We calculate the distante from the depot to the inserted client
            _cost = eoVRPUtils::distance (0, _newClient);

            // And check that its TW is not exceeded
            eoVRPUtils::getTimeWindow (_newClient, readyTime, dueTime, serviceTime);

            if (_cost > dueTime)
                return false;

            // If the vehicle arrives before client's ready time, it has
            // to wait until the client is ready
            if (_cost < readyTime)
                _cost = readyTime;

            // Add the service time for the newly inserted client
            _cost += serviceTime;

            // And the cost of traveling to the next one (the first one in the old route)
            _cost = _cost + eoVRPUtils::distance (_newClient, _route [0]);

        }
        else
            // We just need to add the cost of traveling from the depot to the first client
            _cost = eoVRPUtils::distance (0, _route [0]);

        // Main loop to evaluate the rest of the route (except the last position)
        for (unsigned i = 0; i < _route.size () - 1; i++) {

            // Check that the TW is not exceeded for the current client
            eoVRPUtils::getTimeWindow (_route [i], readyTime, dueTime, serviceTime);

            if (_cost > dueTime)
                return false;

            // If it is not exceeded, we check wether the vehicle arrives before
            // the client is ready. If that's the case, it has to wait
            if (_cost < readyTime)
                _cost = readyTime;

            // We add the service time for this client
            _cost += serviceTime;

            // And now check if we have to insert the new client after the current one
            if (_route [i] == _afterClient) {

                // If that's the case, we add the cost of traveling from current client
                // to the new one
                _cost = _cost + eoVRPUtils::distance (_route [i], _newClient);

                // And check for its TW to be not exceeded
                eoVRPUtils::getTimeWindow (_newClient, readyTime, dueTime, serviceTime);

                if (_cost > dueTime)
                    return false;

                // If the vehicle arrives before client's ready time, it has
                // to wait until the client is ready
                if (_cost < readyTime)
                    _cost = readyTime;

                // Add the service time for the newly inserted client
                _cost += serviceTime;

                // And the cost of traveling to the next one
                _cost = _cost + eoVRPUtils::distance (_newClient, _route [i + 1]);

            }
            else
                // We simply add the cost of traveling to the next client
                _cost = _cost + eoVRPUtils::distance (_route [i], _route [i + 1]);

        }

        // Consider the special case where the new client has
        // to be inserted at the end of the route
        unsigned last =_route [_route.size () - 1];

        // We first check that the TW of the last client in the old route
        // has not been exedeed
        eoVRPUtils::getTimeWindow (last, readyTime, dueTime, serviceTime);

        if (_cost > dueTime)
            return false;

        // If the vehicle arrives before the client is ready, it waits
        if (_cost < readyTime)
            _cost = readyTime;

        // Now we add its service time
        _cost += serviceTime;

        // And check if the new client has to be inserted at the end
        // of the old route
        if (_afterClient == last) {

            // In that case, we add the cost of traveling from the last client
            // to the one being inserted
            _cost = _cost + eoVRPUtils::distance (last, _newClient);

            // Check for its TW not being exceeded
            eoVRPUtils::getTimeWindow (_newClient, readyTime, dueTime, serviceTime);

            if (_cost > dueTime)
                return false;

            // If the vehicle arrives before the new client is ready, it waits
            if (_cost < readyTime)
                _cost = readyTime;

            // Now we add its service time
            _cost += serviceTime;

            // And, finally, the time to travel back to the depot
            _cost = _cost + eoVRPUtils::distance (_newClient, 0);

        }
        else
            // In this case we just add the cost of traveling back to the depot
            _cost = _cost + eoVRPUtils::distance (last, 0);

        // Last thing to check is that the vehicle is back on time to the depot
        eoVRPUtils::getTimeWindow (0, readyTime, dueTime, serviceTime);

        if (_cost > dueTime)
            return false;

        // If all constraints are satisfied, we return true, and the cost of the
        // insertion in the variable "_cost"
        return true;

    }


    /**
      * \brief Selects the farthest client as seed for a new route.
      * \param _unvisited Vector of unvisited and thus available clients for constructing the new route.
      * \param _unvisitedIdx Position of the last univisted client in _unvisited vector.
      * \return The position of the client farthest from the depot.
      */

    unsigned selectFarthestClientAsSeed (const std::vector<int>& _unvisited, int _unvisitedIdx) {

        unsigned maxPos  = 0;
        double   maxDist = eoVRPUtils::distance (0, _unvisited [0]);

        for (unsigned i = 1; i <= _unvisitedIdx; i++)
            if (eoVRPUtils::distance (0, _unvisited [i]) > maxDist) {

                maxPos  = i;
                maxDist = eoVRPUtils::distance (0, _unvisited [i]);

            }

        return maxPos;

    }


    /**
      * \brief Selects the cheapest client as seed for a new route.
      * \param _unvisited Vector of unvisited and thus available clients for constructing the new route.
      * \param _unvisitedIdx Position of the last univisted client in _unvisited vector.
      * \param _seedCheckingOverride If true, it overrides the seed checking mecanism.
      * \return The position of the cheapest client.
      */

    unsigned selectCheapestClient (const std::vector<int>& _unvisited, int _unvisitedIdx, bool _seedCheckingOverride) {

        int    cheapestPos  = -1;
        double cheapestCost = 999999999;

        for (unsigned i = 0; i <= _unvisitedIdx; i++) {

            double cost = (-ALFA * eoVRPUtils::distance (0, _unvisited [i])    ) +
                          ( BETA * eoVRPUtils::clients [_unvisited [i]].dueTime) +
                          (GAMMA * eoVRPUtils::polarAngle (0, _unvisited [i]) / 360 * eoVRPUtils::distance (0, _unvisited [i]));

            if ((cost <  cheapestCost        || (cost == cheapestCost && rng.flip ())) &&
                    (!mSeedsUsed [_unvisited [i]] ||  _seedCheckingOverride               )    ) {

                cheapestPos  = i;
                cheapestCost = cost;

            }

        }

        return cheapestPos;

    }


    /**
      * \brief Selects the best (the "hardest" one) client as seed for a new route.
      * \param _unvisited Vector of unvisited and thus available clients for constructing the new route.
      * \param _unvisitedIdx Position of the last univisted client in _unvisited vector.
      * \param _seedCheckingOverride If true, it overrides the seed checking mecanism.
      * \return The position of the best client.
      */

    unsigned selectBestClientAsSeed (const std::vector<int>& _unvisited, int _unvisitedIdx, bool _seedCheckingOverride) {

        int    cheapestPos  = -1;
        double cheapestCost = 999999999;
        double alfa, beta;

        alfa = rng.uniform ();
        beta = rng.uniform ();

        for (unsigned i = 0; i <= _unvisitedIdx; i++) {

            double cost = (alfa * eoVRPUtils::distance (0, _unvisited [i])) +
                          (beta * (eoVRPUtils::clients [_unvisited [i]].dueTime - eoVRPUtils::clients [_unvisited [i]].readyTime));


            if ((cost <  cheapestCost        || (cost == cheapestCost && rng.flip ())) &&
                    (!mSeedsUsed [_unvisited [i]] ||  _seedCheckingOverride               )    ) {

                cheapestPos  = i;
                cheapestCost = cost;

            }

        }

        return cheapestPos;

    }


    /**
      * \brief Random initializer.
      * Initializes a genotype using a random initializer.
      * @param _gen Generally a genotype that has been default-constructed.
      *             Whatever it contains will be lost.
      */

    void RandomInitializationNoCheck (eoVRP& _gen) {

        // Erase any previous contents
        _gen.clear ();

        // Aux vector to store unvisited clients
        std::vector<int> unvisited;

        // And an index to point to the last unvisited cutomer
        int unvisitedIdx = eoVRPUtils::clients.size () - 2;

        // Initialization of the unvisited vector with all the clients
        for (unsigned i = 1; i < eoVRPUtils::clients.size (); i++)
            unvisited.push_back (i);

        while (unvisitedIdx >= 0) {

            unsigned n = rng.random (unvisitedIdx);

            for (unsigned i = 0; i <= n; i++) {

                unsigned pos = rng.random (unvisitedIdx);

                _gen.push_back (unvisited [pos]);

                unvisited [pos] = unvisited [unvisitedIdx];
                unvisitedIdx--;

            }

        }

    }


};

#endif
