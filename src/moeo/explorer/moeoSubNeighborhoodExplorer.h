/*
* <moeoSubNeighborhoodExplorer.h>
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

#ifndef _MOEOSUBNEIGHBORHOODEXPLORER_H
#define _MOEOSUBNEIGHBORHOODEXPLORER_H

#include "../../eo/eoPop.h"
#include "../../mo/neighborhood/moNeighbor.h"
#include "../../mo/neighborhood/moNeighborhood.h"
#include "moeoPopNeighborhoodExplorer.h"

/**
 * Explorer which explore a part of the neighborhood
 */
template < class Neighbor >
class moeoSubNeighborhoodExplorer : public moeoPopNeighborhoodExplorer < Neighbor >
{
	/** Alias for the type */
    typedef typename Neighbor::EOT MOEOT;
    /** Alias for the objeciveVector */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;

public:

	/**
	 * Constructor
	 * @param _neighborhood a neighborhood
	 * @param _number the number of neighbor to explore
	 */
    moeoSubNeighborhoodExplorer(
    		moNeighborhood<Neighbor>& _neighborhood,
    		unsigned int _number)
            : neighborhood(_neighborhood), number(_number){}

	/**
	 * functor to explore the neighborhood
	 * @param _src the population to explore
	 * @param _select contains index of individuals from the population to explore
	 * @param _dest contains new generated individuals
	 */
    void operator()(eoPop < MOEOT > & _src, std::vector <unsigned int> _select, eoPop < MOEOT > & _dest)
    {
		for(unsigned int i=0; i<_select.size(); i++)
			explore(_src[_select[i]], _dest);
    }

protected:

	/**
	 * explorer of one individual
	 * @param _src the individual to explore
	 * @param _dest contains new generated individuals
	 */
	virtual void explore(MOEOT & _src, eoPop < MOEOT > & _dest) = 0;

	/** Neighbor */
	Neighbor neighbor;
	/** Neighborhood */
	moNeighborhood<Neighbor> & neighborhood;

    /** number of neighbor to explore for each selected individual*/
    unsigned int number;

};

#endif /*_MOEOSUBNEIGHBORHOODEXPLORER_H_*/
