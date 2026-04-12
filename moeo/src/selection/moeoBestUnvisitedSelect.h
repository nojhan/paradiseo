/*
* <moeoBestUnvisitedSelect.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2008
* (C) OPAC Team, LIFL, 2002-2008
*
* Arnaud Liefooghe
* Jérémie Humeau
* Eremey Valetov
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

#ifndef _MOEOBESTUNVISITEDSELECT_H
#define _MOEOBESTUNVISITEDSELECT_H

#include <eoPop.h>
#include <selection/moeoUnvisitedSelect.h>
#include <comparator/moeoParetoObjectiveVectorComparator.h>
#include <algorithm>
#include <random>

/**
 * Selects up to a given number of unvisited (or all) individuals from a population,
 * optionally excluding those dominated by given objective vectors. The selected
 * indices are returned in shuffled order.
 */
template < class MOEOT >
class moeoBestUnvisitedSelect : public moeoUnvisitedSelect < MOEOT >
{
public:
    /**
     * @param _number maximum number of individuals to select
     * @param _selectVisited if true, also consider visited individuals
     */
    moeoBestUnvisitedSelect(unsigned int _number, bool _selectVisited = true)
        : number(_number), selectVisited(_selectVisited) {}

    /**
     * @param _number maximum number of individuals to select
     * @param _excludeObjectives individuals dominated by any of these are excluded
     * @param _selectVisited if true, also consider visited individuals
     */
    moeoBestUnvisitedSelect(unsigned int _number,
                            const std::vector<typename MOEOT::ObjectiveVector>& _excludeObjectives,
                            bool _selectVisited = true)
        : number(_number), excludeObjectives(_excludeObjectives), selectVisited(_selectVisited) {}

    std::vector<unsigned int> operator()(eoPop<MOEOT>& _src) override
    {
        std::vector<unsigned int> res;
        res.reserve(_src.size());

        moeoParetoObjectiveVectorComparator<typename MOEOT::ObjectiveVector> comparator;

        for (unsigned int i = 0; i < _src.size(); i++) {
            if (!selectVisited && _src[i].flag() != 0)
                continue;

            bool isExcluded = false;
            for (const auto& objVec : excludeObjectives) {
                if (comparator(_src[i].objectiveVector(), objVec)) {
                    isExcluded = true;
                    eo::log << eo::debug << "Excluding individual " << i
                            << " based on exclusion objective vector" << std::endl;
                    break;
                }
            }
            if (!isExcluded)
                res.push_back(i);
        }

        // Ensure at least one unvisited individual if possible
        if (!selectVisited && res.empty()) {
            for (unsigned int i = 0; i < _src.size(); i++) {
                if (_src[i].flag() == 0) {
                    res.push_back(i);
                    break;
                }
            }
        }

        // Last resort: pick the first individual
        if (res.empty() && !_src.empty())
            res.push_back(0);

        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(res.begin(), res.end(), g);

        res.resize(std::min(number, static_cast<unsigned int>(res.size())));
        return res;
    }

private:
    unsigned int number;
    std::vector<typename MOEOT::ObjectiveVector> excludeObjectives;
    bool selectVisited;
};

#endif /*_MOEOBESTUNVISITEDSELECT_H*/
