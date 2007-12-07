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

#ifndef eoVRPQuadCrossover_H
#define eoVRPQuadCrossover_H

// General includes
#include <assert.h>
#include <values.h>
#include <utils/eoRNG.h>
#include <set>

// The base definition of eoQuadOp
#include <eoOp.h>

/**
  * \class eoVRPGenericCrossover eoVRPQuadCrossover.h
  * \brief Implementation of the generic crossover for the VRP-TW by Tavares et al.
  */

class eoVRPGenericCrossover: public eoQuadOp <eoVRP> {

public:

    /**
      * \brief Deafult constructor.
      */

    eoVRPGenericCrossover () {

    }


    /**
      * \brief Returns a string containing the name of the class. Used to display statistics.
      * \return The string containing the name of the class.
      */

    std::string className () const {

        return "eoVRPGenericCrossover";

    }


    /**
      * \brief Both parameters are the parents and the (future) children of the crossover.
      * \param _genotype1 The first parent.
      * \param _genotype2 The second parent.
      * \return True if any of the parents was modified. False otherwise.
      */

    bool operator () (eoVRP& _genotype1, eoVRP& _genotype2) {

        Routes c1 = _genotype1.routes ();
        Routes c2 = _genotype2.routes ();

        GenericCrossover (_genotype1.routes (), c2);
        GenericCrossover (_genotype2.routes (), c1);

        _genotype1.encode (c1);
        _genotype2.encode (c2);

        return true;

    }


private:

    /**
      * \brief Actually performs the generic crossover.
      * \param _donor Set of routes from the first parent.
      * \param _receiver Set of routes from the second parent
      * \return True if the second parent was modified. False otherwise.
      */

    bool GenericCrossover (const Routes& _donor, Routes& _receiver) const {

        unsigned srcRoute = rng.random (_donor.size ());
        unsigned srcPos1  = rng.random (_donor [srcRoute].size ());
        unsigned srcPos2  = rng.random (_donor [srcRoute].size ());

        if (srcPos1 > srcPos2)
            std::swap (srcPos1, srcPos2);

        Route::iterator it;

        for (unsigned i = srcPos1; i <= srcPos2; i++)
            for (unsigned j = 0; j < _receiver.size (); j++) {

                it = find (_receiver [j].begin (), _receiver [j].end (), _donor [srcRoute][i]);

                if (it != _receiver [j].end ()) {

                    // Deletion of the repeated client
                    _receiver [j].erase (it);

                    // Deletion of empty route, if necessary
                    if (_receiver [j].size () == 0)
                        _receiver.erase (_receiver.begin () + j);

                    break;

                }

            }

        unsigned dstRoute = rng.random (_receiver.size ());

        it = _receiver [dstRoute].begin () + rng.random (_receiver [dstRoute].size ());

        _receiver [dstRoute].insert (it + 1, _donor [srcRoute].begin () + srcPos1, _donor [srcRoute].begin () + srcPos2 + 1);

        return true;

    }

};


/**
  * \class eoVRPOnePointCrossover eoVRPQuadCrossover.h
  * \brief Implementation of the simple One Point Crossover.
  */

class eoVRPOnePointCrossover: public eoQuadOp <eoVRP> {

public:

    /**
      * \brief Deafult constructor.
      */

    eoVRPOnePointCrossover () {

    }


    /**
      * \brief Returns a string containing the name of the class. Used to display statistics.
      * \return The string containing the name of the class.
      */

    std::string className () const {

        return "eoVRPOnePointCrossover";

    }


    /**
      * \brief Performs a one point crossover. Both parameters are the parents and the (future) children of the crossover.
      * \param _genotype1 The first parent.
      * \param _genotype2 The second parent.
      * \return True if any of the parents was modified. False otherwise.
      */

    bool operator () (eoVRP& _genotype1, eoVRP& _genotype2) {

        eoVRP& _gen = _genotype1;

        unsigned orig1, orig2, dest;

        // First child
        orig1 = rng.random (_genotype2.size ());
        orig2 = rng.random (_genotype2.size ());

        if (orig1 > orig2)
            std::swap (orig1, orig2);

        for (unsigned i = orig1; i <= orig2; i++)
            _genotype1.erase (find (_genotype1.begin (), _genotype1.end (), _genotype2 [i]));

        dest = rng.random (_genotype1.size ());

        _genotype1.insert (_genotype1.begin () + dest, _genotype2.begin () + orig1, _genotype2.begin () + orig2 + 1);

        // Second child
        orig1 = rng.random (_gen.size ());
        orig2 = rng.random (_gen.size ());

        if (orig1 > orig2)
            std::swap (orig1, orig2);

        for (unsigned i = orig1; i <= orig2; i++)
            _genotype2.erase (find (_genotype2.begin (), _genotype2.end (), _gen [i]));

        dest = rng.random (_genotype2.size ());

        _genotype2.insert (_genotype2.begin () + dest, _gen.begin () + orig1, _gen.begin () + orig2 + 1);

        _genotype1.cleanRoutes ();
        _genotype2.cleanRoutes ();

        return true;

    }

};


/**
  * \class eoVRPEdgeCrossover eoVRPQuadCrossover.h
  * \brief Implementation of the classic Edge Crossover from the TSP.
  */

class eoVRPEdgeCrossover: public eoQuadOp <eoVRP> {

public:

    /**
      * \brief Deafult constructor.
      */

    eoVRPEdgeCrossover () {

    }


    /**
      * \brief Returns a string containing the name of the class. Used to display statistics.
      * \return The string containing the name of the class.
      */

    std::string className () const {

        return "eoVRPEdgeCrossover";

    }


    /**
      * \brief Both parameters are the parents and the (future) children of the crossover.
      * \param _genotype1 The first parent.
      * \param _genotype2 The second parent.
      * \return True if any of the parents was modified. False otherwise.
      */

    bool operator () (eoVRP& _genotype1, eoVRP& _genotype2) {

        eoVRP par [2];

        // Backup of the parents
        par [0] = _genotype1;
        par [1] = _genotype2;

        _genotype1.clean ();
        _genotype2.clean ();

        EdgeCrossover (par [0], par [1], _genotype1);
        EdgeCrossover (par [0], par [1], _genotype2);

        return true;

    }


private:

    /**
      * \brief Actually performs the edge crossover.
      * \param _genotype1 First parent.
      * \param _genotype2 Second parent.
      * \param _child Child.
      * \return True if the second parent was modified. False otherwise.
      */

    bool EdgeCrossover (eoVRP& _genotype1, eoVRP& _genotype2, eoVRP& _child) {

        std::vector <std::set <unsigned> > _map;
        std::vector <bool> visited;

        // Build map
        unsigned len = _genotype1.size () ;

        _map.resize (len+1) ;

        for (unsigned i = 0 ; i < len ; i ++) {

            _map [_genotype1 [i]].insert (_genotype1 [(i + 1) % len]) ;
            _map [_genotype2 [i]].insert (_genotype2 [(i + 1) % len]) ;
            _map [_genotype1 [i]].insert (_genotype1 [(i - 1 + len) % len]) ;
            _map [_genotype2 [i]].insert (_genotype2 [(i - 1 + len) % len]) ;

        }

        visited.clear () ;
        visited.resize (len+1, false) ;


        _child.clear () ;

        unsigned cur_vertex = rng.random (len)+1;

        add_vertex (cur_vertex, visited, _map, _child);

        for (unsigned i = 1; i < len; i ++) {

            unsigned len_min_entry = MAXINT;

            std::set <unsigned>& neigh = _map [cur_vertex];

            for (std::set <unsigned>::iterator it = neigh.begin (); it != neigh.end (); it ++) {

                    unsigned l = _map [*it].size ();

                    if (len_min_entry > l)
                        len_min_entry = l;

                }

            std::vector <unsigned> cand; /* Candidates */

            for (std::set <unsigned>::iterator it = neigh.begin (); it != neigh.end (); it ++) {

                    unsigned l = _map [*it].size ();

                    if (len_min_entry == l)
                        cand.push_back (*it);

                }

            if (!cand.size ()) {

                /* Oh no ! Implicit mutation */
                for (unsigned j = 1; j <= len; j ++)
                    if (!visited [j])
                        cand.push_back (j);

            }

            cur_vertex = cand [rng.random (cand.size ())] ;

            add_vertex (cur_vertex, visited, _map, _child);

        }

    }


    /**
      * \brief Removes a vertex from all his neighbours.
      * \param _vertex The vertex being erased.
      * \param _map The structure containing the neighbourhood relationship.
      */

    void remove_entry (unsigned _vertex, std::vector <std::set <unsigned> >& _map) {

        std::set <unsigned>& neigh = _map [_vertex];

        for (std::set <unsigned>::iterator it = neigh.begin (); it != neigh.end (); it++)
                _map [*it].erase (_vertex);

    }


    /**
      * \brief Adds a vertex to a child and erases it from the list of available vertices.
      * \param _vertex The vertex being added to the child.
      * \param _visited The vector of visited vertices.
      * \param _map The structure containing the neighbourhood relationship.
      * \param _child The child where we add the vertex.
      */

    void add_vertex (unsigned _vertex, std::vector <bool>& _visited, std::vector <std::set <unsigned> >& _map, eoVRP& _child) {

        _visited [_vertex] = true;
        _child.push_back (_vertex);
        remove_entry (_vertex, _map);

    }

};

#endif
