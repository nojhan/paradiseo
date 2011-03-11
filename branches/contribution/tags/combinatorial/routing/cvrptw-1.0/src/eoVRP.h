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

#ifndef _eoVRP_h
#define _eoVRP_h

// The base definition of eoVector
#include <eoVector.h>

// Utilities for the VRP-TW problem
#include "eoVRPUtils.h"

/**
  * \class eoVRP eoVRP.h
  * \brief Defines the getoype used to solve the VRP-TW problem.
  */

class eoVRP: public eoVector<eoMinimizingFitness, int> {

public:

    /**
      * \brief Default constructor: initializes variables to safe values.
      */

    eoVRP () : mLength (0.0) {

    }


    /**
      * \brief Copy contructor: creates a new individual from a given one.
      * \param _orig The individual used to create the new one.
      */

    eoVRP (const eoVRP& _orig) {

        operator= (_orig);

    }


    /**
      * \brief Default destructor: nothing to do here.
      */

    virtual ~eoVRP () {

    }


    /**
      * \brief Performs a copy from the invidual passed as argument.
      * \param _orig The individual to copy from.
      * \return A reference to this.
      */

    eoVRP& operator= (const eoVRP& _orig) {

        // Sanity check
        if (&_orig != this) {

            // Cleans both individual and decoding information
            clean ();

            // We call the assignment operator from the base class
            eoVector<eoMinimizingFitness, int>::operator= (_orig);

            // And then copy all our attributes
            mRoutes  = _orig.mRoutes;
            mLength  = _orig.mLength;

        }

        return *this;

    }


    /**
      * \brief Returns a string containing the name of the class.
      * \return The string containing the name of the class.
      */

    virtual std::string className () const {

        return "eoVRP";

    }


    /**
      * \brief Prints the individual to a given stream.
      * \param _os The stream to print to.
      */

    void printOn (std::ostream& _os) const {

        // First write the fitness
        _os << std::endl;

        // Then the individual itself using the base printing method
        eoVector<eoMinimizingFitness, int>::printOn (_os);
        _os << std::endl << std::endl;

    }


    /**
      * \brief Prints a detailed version of the individual (decoding information, unsatisfied contraints, etc.) to a given stream.
      * \param _os The stream to print to.
      */

    void printAllOn (std::ostream& _os) const {

        // Print the individual itself using the base printing method
        eoVector<eoMinimizingFitness, int>::printOn (_os);
        _os << std::endl << std::endl;

        // Check if we have decoding information to print
        if (decoded ()) {

            // First, we print the decoded routes (stored in mRoutes)
            _os << " => Routes: " << std::endl << std::endl;
            printRoutes (_os);
            _os << std::endl << std::endl;

            if (this->invalid ())
                _os << " => Fitness: INVALID." << std::endl << std::endl;
            else
                _os << " => Fitness: " << this->fitness () << std::endl << std::endl;

        }
        else
            std::cerr << "Warning: 'printAllOn' called but the individual was not already decoded." << std::endl;

    }


    /**
      * \brief Reads an individual from a given stream.
      * \param _is The stream to read from.
      */

    void readFrom (std::istream& _is) {

        // Read the individual using the method from the base class
        eoVector<eoMinimizingFitness, int>::readFrom (_is);

    }


    /**
      * \brief Returns a reference to the decoded individual.
      * \return A reference to the decoded individual.
      */

    const Routes& routes () {

        if (mRoutes.size () == 0)
            std::cerr << "Warning: This individual has not been already decoded." << std::endl;

        return mRoutes;

    }


    /**
      * \brief Returns the total cost (length) of traveling all the routes.
      * \return The total cost (length) of traveling all the routes.
      */

    double length () {

        return mLength;

    }


    /**
      * \brief Aux. method to print a structure of routes.
      * \param _os The stream to print to.
      */

    void printRoutes (std::ostream& _os) const {

        _os << "[";

        for (unsigned i = 0; i < mRoutes.size (); i++) {

            _os << "[";

            printRoute (_os, i);

            if (i == mRoutes.size () - 1)
                _os << "]";
            else
                _os << "]," << std::endl;
        }

        _os << "]";

    }


    /**
      * \brief Aux. method to print only one route.
      * \param _os The stream to print to.
      * \param _p The route to print.
      */

    void printRoute (std::ostream& _os, unsigned _p) const {

        _os << "[";

        for (unsigned i = 0; i < mRoutes [_p].size (); i++) {

            _os << mRoutes [_p][i];

            if (i != mRoutes [_p].size () - 1)
                _os << ", ";

        }

        _os << "]";

    }


    /**
      * \brief Cleans the individual (the vector of clients and also the decoding information).
      * \return True if the operation finishes correctly. False otherwise.
      */

    bool clean () {

        this->clear ();
        mRoutes.clear ();
        mLength = 0.0;

        return true;

    }


    /**
      * \brief Invalidates the decoding information (usually after crossover or mutation).
      * \return True if the operation finishes correctly. False otherwise.
      */

    bool cleanRoutes () {

        mRoutes.clear ();
        mLength = 0.0;

        return true;

    }


    /**
      * \brief Has this individual been decoded?
      * \return True if has decoding information. False otherwise.
      */

    bool decoded () const {

        if (mRoutes.size () == 0)
            return false;

        return true;

    }


    /**
      * \brief Encodes an individual from a set of routes (usually used within crossover).
      * \return True if the operation finishes correctly. False otherwise.
      */

    bool encode (Routes& _routes) {

        clean ();

        for (unsigned i = 0; i < _routes.size (); i++) {

            for (unsigned j = 0; j < _routes [i].size (); j++)
                this->push_back (_routes [i][j]);

        }

        return true;

    }


    /**
      * \brief Decodes an individual in a set of routes and calculates its cost (length) of traveling.
      * \return The cost (length) of traveling the set of routes.
      */

    double decode () {

        bool routeStart = true;

        double demand = 0.0, route_len = 0.0, time = 0.0;
        double readyTime, dueTime, serviceTime;
        double depotReadyTime, depotDueTime, depotServiceTime;

        cleanRoutes ();

        Route route;

        eoVRPUtils::getTimeWindow (0, depotReadyTime, depotDueTime, depotServiceTime);

        for (unsigned i = 0; i < this->size (); i++) {

            if (routeStart) {

                demand = eoVRPUtils::clients [this->operator[] (i)].demand;
                route_len = eoVRPUtils::distance (0, this->operator[] (i));
                time = eoVRPUtils::distance (0, this->operator[] (i));

                // The capacity of the vehicle must NEVER be exceeded by the first client
                // (it would be an instance impossible to solve in that case)
                if (demand > VEHICLE_CAPACITY) {

                    std::cerr << "This should never happen: " << std::endl;
                    abort ();

                }

                // Check that its TW is not exceeded
                eoVRPUtils::getTimeWindow (this->operator[] (i), readyTime, dueTime, serviceTime);

                // Same thing as with capacity and first client, but now with the TW
                if (time > dueTime) {

                    std::cerr << "This should never happen: " << std::endl;
                    abort ();

                }
                else if (time < readyTime)
                    time = readyTime;

                time += serviceTime;

                route.push_back (this->operator[] (i));

                routeStart = false;

            }
            else {

                time += eoVRPUtils::distance (this->operator[] (i - 1), this->operator[] (i));

                // Check that its TW is not exceeded
                eoVRPUtils::getTimeWindow (this->operator[] (i), readyTime, dueTime, serviceTime);

                if ((time > dueTime) || (time + serviceTime + eoVRPUtils::distance (this->operator[] (i), 0) > depotDueTime) ||
                        (demand + eoVRPUtils::clients [this->operator[] (i)].demand > VEHICLE_CAPACITY)                             ) {

                    route_len += eoVRPUtils::distance (this->operator[] (i - 1), 0);

                    mLength += route_len;

                    i--;
                    routeStart = true;

                    // A route should contain, at least, one client. This should never happen, anyway...
                    if (route.size () == 0) {

                        std::cerr << "Error: empty route detected while decoding. The wrong genome follows..." << std::endl;
                        this->printOn (std::cerr);
                        abort ();

                    }

                    mRoutes.push_back (route);
                    route.clear ();

                }
                else {

                    if (time < readyTime)
                        time = readyTime;

                    time += serviceTime;

                    route_len += eoVRPUtils::distance (this->operator[] (i - 1), this->operator[] (i));
                    demand += eoVRPUtils::clients [this->operator[] (i)].demand;

                    route.push_back (this->operator[] (i));

                }

            }

        }

        if (route.size () > 0) {

            route_len += eoVRPUtils::distance (route [route.size () - 1], 0);

            mLength += route_len;
            mRoutes.push_back (route);
            route.clear ();

        }

        return mLength;

    }


private:

    Routes mRoutes;   /**< A set of routes containing the decoding information of the individual. */
    double mLength;   /**< Cached cost (length) of traveling the set of routes defined by the individual. */

};

#endif
