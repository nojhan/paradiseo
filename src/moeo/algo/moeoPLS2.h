/*
* <moeoPLS2.h>
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

#ifndef _MOEOPLS2_H
#define _MOEOPLS2_H

#include "moeoUnifiedDominanceBasedLS.h"
#include "../selection/moeoExhaustiveUnvisitedSelect.h"
#include "../explorer/moeoExhaustiveNeighborhoodExplorer.h"

/**
 * PLS2 algorithm
 *
 * Talbi EG, Rahoual M, Mabed MH, Dhaenens C (2001) A hybrid evolutionary approach for
 * multicriteria optimization problems : Application to the fow shop. In: First International
 * Conference on Evolutionary Multi-criterion Optimization (EMO 2001), Springer-Verlag,
 * Zurich, Switzerland, Lecture Notes in Computer Science, vol 1993, pp 416–428
 *
 * Basseur M, Seynhaeve F, Talbi E (2003) Adaptive mechanisms for multiobjective evolution-
 * ary algorithms. In: Congress on Engineering in System Application (CESA 2003), Lille,
 * France, pp 72–86
 *
 * Angel E, Bampis E, Gourv ́ s L (2004) A dynasearch neighbohood for the bicriteria travel-
 * ing salesman problem. In: Metaheuristics for Multiobjective Optimisation, Lecture Notes
 * in Economics and Mathematical Systems, vol 535, Springer-Verlag, Berlin, Germany,
 * chap 6, pp 153–176
 */
template < class Neighbor >
class moeoPLS2 : public moeoUnifiedDominanceBasedLS < Neighbor >
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
    moeoPLS2(
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
            			*(new moeoExhaustiveUnvisitedSelect<MOEOT>())
            	){}
    
};

#endif /*MOEOPLS2_H_*/
