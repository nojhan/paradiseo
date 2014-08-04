/*
* <moeoPLS1.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2008
* (C) OPAC Team, LIFL, 2002-2008
*
* Arnaud Liefooghe
* Jérémie Humeau
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
//-----------------------------------------------------------------------------

#ifndef _MOEOPLS1_H
#define _MOEOPLS1_H

#include "moeoUnifiedDominanceBasedLS.h"
#include "../selection/moeoNumberUnvisitedSelect.h"
#include "../explorer/moeoExhaustiveNeighborhoodExplorer.h"

/**
 * PLS1 algorithm
 *
 * Paquete L, Chiarandini M, St ̈ tzle T (2004) Pareto local optimum sets in the biobjective
 * traveling salesman problem: An experimental study. In: Metaheuristics for Multiobjective
 * Optimisation, Lecture Notes in Economics and Mathematical Systems, vol 535, Springer-
 * Verlag, Berlin, Germany, chap 7, pp 177–199
 */
template < class Neighbor >
class moeoPLS1 : public moeoUnifiedDominanceBasedLS < Neighbor >
{
public:

	/** Alias for the type */
    typedef typename Neighbor::EOT MOEOT;
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;
	/**
	 * Ctor
	 * @param _continuator a stop creterion
	 * @param _eval a evaluation function
	 * @param _archive a archive to store no-dominated individuals
	 * @param _neighborhood a neighborhood
	 * @param _incrEval neighbor evaluation function
	 */
    moeoPLS1(
    		eoContinue < MOEOT > & _continuator,
            eoEvalFunc < MOEOT > & _eval,
            moeoArchive < MOEOT > & _archive,
    		moNeighborhood<Neighbor>& _neighborhood,
        	moEval < Neighbor > & _incrEval):
            	moeoUnifiedDominanceBasedLS<Neighbor>(
            			_continuator,
            			_eval,
            			_archive,
            			*(new moeoExhaustiveNeighborhoodExplorer<Neighbor>(_neighborhood, _incrEval)),
            			*(new moeoNumberUnvisitedSelect<MOEOT>(1))
            	){}

    
};

#endif /*MOEOPLS1_H_*/
