/*
* <moeoUnifiedDominanceBasedLS_Real.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2008
* (C) OPAC Team, LIFL, 2002-2008
*
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

#ifndef _MOEOUNIFIEDDOMINANCEBASEDLSREAL_H
#define _MOEOUNIFIEDDOMINANCEBASEDLSREAL_H

#include <eoPop.h>
#include <eoContinue.h>
#include <eoEvalFunc.h>
#include <utils/eoLogger.h>
#include <archive/moeoArchive.h>
#include <algo/moeoPopLS.h>
#include <explorer/moeoPopNeighborhoodExplorer.h>
#include <selection/moeoUnvisitedSelect.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * Real-valued dominance-based local search for multi-objective optimization.
 * Uses an archive, flexible selection strategy, and neighborhood exploration
 * with optional OpenMP parallelism for initial population evaluation.
 */
template < class Neighbor >
class moeoUnifiedDominanceBasedLSReal : public moeoPopLS < Neighbor >
{
public:
    typedef typename Neighbor::EOT MOEOT;

    moeoUnifiedDominanceBasedLSReal(
        eoContinue < MOEOT > & _continuator,
        eoEvalFunc < MOEOT > & _eval,
        moeoArchive < MOEOT > & _archive,
        moeoPopNeighborhoodExplorer < Neighbor > & _explorer,
        moeoUnvisitedSelect < MOEOT > & _select)
        : continuator(_continuator), eval(_eval),
          archive(_archive), explorer(_explorer), select(_select) {}

    virtual void operator()(eoPop<MOEOT>& _pop) override
    {
        size_t initialPopSize = _pop.size();
        int generation = 0;

        std::vector<unsigned int> selectionVector;
        std::vector<int> initialFlags;
        eoPop<MOEOT> tmp_pop;

#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (size_t i = 0; i < _pop.size(); i++)
            eval(_pop[i]);

        archive(_pop);

        do {
            tmp_pop.resize(0);
            selectionVector = select(archive);

            initialFlags.clear();
            initialFlags.reserve(selectionVector.size());
            for (auto idx : selectionVector)
                initialFlags.push_back(archive[idx].flag());

            eo::log << eo::progress << "LS generation " << ++generation
                    << ", archive size = " << archive.size() << std::endl;
            eo::log << eo::debug << "continuator = " << continuator << std::endl;

            explorer(archive, selectionVector, tmp_pop);

            // Verify explorer incremented visit counts correctly
            bool allIncremented = true;
            bool noneIncremented = true;
            for (size_t i = 0; i < selectionVector.size(); i++) {
                int newFlag = archive[selectionVector[i]].flag();
                if (newFlag == initialFlags[i] + 1)
                    noneIncremented = false;
                else
                    allIncremented = false;
            }

            if (!allIncremented && noneIncremented) {
                eo::log << eo::warnings << "Explorer did not increment exploration counts" << std::endl;
                for (auto idx : selectionVector)
                    archive[idx].flag(archive[idx].flag() + 1);
            } else if (!allIncremented) {
                eo::log << eo::warnings << "Exploration counts incremented inconsistently" << std::endl;
                for (size_t i = 0; i < selectionVector.size(); i++)
                    archive[selectionVector[i]].flag(initialFlags[i] + 1);
            }

            archive(tmp_pop);
            archive(_pop);

            if (archive.size() <= initialPopSize) {
                _pop = archive;
            } else {
                tmp_pop.resize(0);
                for (size_t i = 0; i < initialPopSize; i++)
                    tmp_pop.push_back(archive[i]);
                _pop = tmp_pop;
            }
        } while (continuator(archive));
    }

protected:
    eoContinue < MOEOT > & continuator;
    eoEvalFunc<MOEOT>& eval;
    moeoArchive < MOEOT > & archive;
    moeoPopNeighborhoodExplorer < Neighbor > & explorer;
    moeoUnvisitedSelect < MOEOT > & select;
};

#endif /* _MOEOUNIFIEDDOMINANCEBASEDLSREAL_H */
