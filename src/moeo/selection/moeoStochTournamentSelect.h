/*
* <moeoStochTournamentSelect.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Arnaud Liefooghe
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

#ifndef MOEOSTOCHTOURNAMENTSELECT_H_
#define MOEOSTOCHTOURNAMENTSELECT_H_

#include "../comparator/moeoComparator.h"
#include "../comparator/moeoFitnessThenDiversityComparator.h"
#include "moeoSelectOne.h"
#include "moeoSelectors.h"

/**
 *  Selection strategy that selects ONE individual by stochastic tournament.
 */
template < class MOEOT > class moeoStochTournamentSelect:public moeoSelectOne <MOEOT>
  {
  public:

    /**
     * Full Ctor
     * @param _comparator the comparator (used to compare 2 individuals)
     * @param _tRate the tournament rate
     */
    moeoStochTournamentSelect (moeoComparator < MOEOT > & _comparator, double _tRate = 1.0) : comparator (_comparator), tRate (_tRate)
    {
      // consistency checks
      if (tRate < 0.5)
        {
          std::cerr << "Warning, Tournament rate should be > 0.5\nAdjusted to 0.55\n";
          tRate = 0.55;
        }
      if (tRate > 1)
        {
          std::cerr << "Warning, Tournament rate should be < 1\nAdjusted to 1\n";
          tRate = 1;
        }
    }


    /**
     * Ctor without comparator. A moeoFitnessThenDiversityComparator is used as default.
     * @param _tRate the tournament rate
     */
    moeoStochTournamentSelect (double _tRate = 1.0) : comparator (defaultComparator), tRate (_tRate)
    {
      // consistency checks
      if (tRate < 0.5)
        {
          std::cerr << "Warning, Tournament rate should be > 0.5\nAdjusted to 0.55\n";
          tRate = 0.55;
        }
      if (tRate > 1)
        {
          std::cerr << "Warning, Tournament rate should be < 1\nAdjusted to 1\n";
          tRate = 1;
        }
    }


    /**
     *  Apply the tournament to the given population
     * @param _pop the population
     */
    const MOEOT & operator() (const eoPop < MOEOT > &_pop)
    {
      // use the selector
      return mo_stochastic_tournament(_pop,tRate,comparator);
    }


  protected:

    /** the comparator (used to compare 2 individuals) */
    moeoComparator < MOEOT > & comparator;
    /** a fitness then diversity comparator can be used as default */
    moeoFitnessThenDiversityComparator < MOEOT > defaultComparator;
    /** the tournament rate */
    double tRate;

  };

#endif /*MOEOSTOCHTOURNAMENTSELECT_H_ */
