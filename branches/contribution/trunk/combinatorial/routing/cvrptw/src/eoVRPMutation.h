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

#ifndef eoVRPMutation_H
#define eoVRPMutation_H

// General includes
#include <algorithm>

// The base definition of eoMonOp
#include <eoOp.h>

/**
  * \class eoVRPMutation eoVRPMutation.h
  * \brief Implementation of variations of the four mutation operators for the VRP-TW defined by Tavares et al.
  * These four operators should be separated in different classes and their probabilities
  * made parameterizable.
  */

class eoVRPMutation: public eoMonOp <eoVRP> {

public:

    /**
      * \brief Default constructor: nothing to do here.
      */

    eoVRPMutation () {

    }


    /**
      * \brief Returns a string containing the name of the class. Used to display statistics.
      * \return The string containing the name of the class.
      */

    std::string className () const {

        return "eoVRPMutation";

    }


    /**
      * \brief Functor operator.
      * Applies one of the four mutation operators available, each of them with a predefined
      * (hard-coded) probability. These operators should be separated in different classes
      * and their probabilities made parameterizable to do it in a more "paradisEO" way.
      * \param _genotype The genotype being mutated (it will be probably modified).
      * \return True if the individual has been modified. False otherwise.
      */

    bool operator () (eoVRP& _genotype) {

        bool   res = false;
        double op  = rng.uniform ();


        if (op <= 0.05)
            res = swapMutation (_genotype);
        else if ((op > 0.05) && (op <= 0.20))
            res = inversionMutation (_genotype);
        else if ((op > 0.20) && (op <= 0.25))
            res = insertionMutation (_genotype);
        else if ((op > 0.25) && (op <= 0.45))
            res = DisplacementMutation (_genotype);

        if (res)
            _genotype.cleanRoutes ();

        return res;

    }


private:


    /**
      * \brief It exhanges the positions of two clients within the individual.
      * Clients may or may not be in the same route.
      * \param _genotype The genotype being mutated (it will be probably modified).
      * \return True if the individual has been modified. False otherwise.
      */

    bool swapMutation (eoVRP& _genotype) {

        int p1 = rng.random (_genotype.size ());
        int p2 = -1;

        do {
            p2 = rng.random (_genotype.size ());
        } while (_genotype [p2] == _genotype [p1]);

        std::swap (_genotype [p1], _genotype [p2]);

        return true;

    }


    /**
      * \brief It selects two positions in the genotype and inverts the clients between them.
      * Clients may or may not be in the same route.
      * \param _genotype The genotype being mutated (it will be probably modified).
      * \return True if the individual has been modified. False otherwise.
      */

    bool inversionMutation (eoVRP& _genotype) {

        int p1 = rng.random (_genotype.size ());
        int p2 = -1;

        do {
            p2 = rng.random (_genotype.size ());
        } while (_genotype [p2] == _genotype [p1]);

        if (p1 > p2)
            std::swap (p1, p2);

        // Reverse the subroute
        reverse (_genotype.begin () + p1, _genotype.begin () + p2 + 1);


        return false;

    }


    /**
      * \brief It selects and individual, erases it from its original position and inserts it somewhere else.
      * The insertion may or may not be within the same route.
      * \param _genotype The genotype being mutated (it will be probably modified).
      * \return True if the individual has been modified. False otherwise.
      */

    bool insertionMutation (eoVRP& _genotype) {

        int p = -1;

        // Selection of the client to be moved
        do {
            p = rng.random (_genotype.size ());
        } while (_genotype [p] == -1);

        // Temporary copy of the client
        unsigned client = _genotype [p];

        _genotype.erase (_genotype.begin () + p);

        p = rng.random (_genotype.size () - 1);
        _genotype.insert (_genotype.begin () + p, client);

        return true;

    }


    /**
      * \brief It selects a set of clients, erases them from their original position and inserts them somewhere else.
      * The selected set of clients may cover different routes.
      * \param _genotype The genotype being mutated (it will be probably modified).
      * \return True if the individual has been modified. False otherwise.
      */

    bool DisplacementMutation (eoVRP& _genotype) {

        int p1 = rng.random (_genotype.size ());
        int p2 = -1;

        do {
            p2 = rng.random (_genotype.size ());
        } while (_genotype [p2] == _genotype [p1]);

        if (p1 > p2)
            std::swap (p1, p2);

        // Temporary copy of the fragment being moved
        Route route;

        for (unsigned i = p1; i <= p2; i++)
            route.push_back (_genotype [i]);

        _genotype.erase (_genotype.begin () + p1, _genotype.begin () + p2 + 1);

        unsigned p = rng.random ((_genotype.size () > 0) ? _genotype.size () - 1 : 0);
        _genotype.insert (_genotype.begin () + p, route.begin (), route.end ());

        return true;

    }


};

#endif
