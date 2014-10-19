/*
* <moeoBinaryMetricSavingUpdater.h>
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

#ifndef MOEOBINARYMETRICSAVINGUPDATER_H_
#define MOEOBINARYMETRICSAVINGUPDATER_H_

#include <fstream>
#include <string>
#include <vector>
#include "../../eo/eoPop.h"
#include "../../eo/utils/eoUpdater.h"
#include "../metric/moeoMetric.h"

/**
 * This class allows to save the progression of a binary metric comparing the objective vectors of the current population (or archive)
 * with the objective vectors of the population (or archive) of the generation (n-1) into a file
 */
template < class MOEOT >
class moeoBinaryMetricSavingUpdater : public eoUpdater
  {
  public:

    /** The objective vector type of a solution */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;


    /**
     * Ctor
     * @param _metric the binary metric comparing two Pareto sets
     * @param _pop the main population
     * @param _filename the target filename
     */
    moeoBinaryMetricSavingUpdater (moeoVectorVsVectorBinaryMetric < ObjectiveVector, double > & _metric, const eoPop < MOEOT > & _pop, std::string _filename) :
        metric(_metric), pop(_pop), filename(_filename), counter(1), firstGen(true)
    {}


    /**
     * Saves the metric's value for the current generation
     */
    void operator()()
    {
      if (pop.size())
        {
          if (firstGen)
            {
              firstGen = false;
            }
          else
            {
              // creation of the two Pareto sets
              std::vector < ObjectiveVector > from;
              std::vector < ObjectiveVector > to;
              for (unsigned int i=0; i<pop.size(); i++)
                from.push_back(pop[i].objectiveVector());
              for (unsigned int i=0 ; i<oldPop.size(); i++)
                to.push_back(oldPop[i].objectiveVector());
              // writing the result into the file
              std::ofstream f (filename.c_str(), std::ios::app);
              f << counter++ << ' ' << metric(from,to) << std::endl;
              f.close();
            }
          oldPop = pop;
        }
    }


  private:

    /** binary metric comparing two Pareto sets */
    moeoVectorVsVectorBinaryMetric < ObjectiveVector, double > & metric;
    /** main population */
    const eoPop < MOEOT > & pop;
    /** (n-1) population */
    eoPop< MOEOT > oldPop;
    /** target filename */
    std::string filename;
    /** is it the first generation ? */
    bool firstGen;
    /** counter */
    unsigned int counter;

  };

#endif /*MOEOBINARYMETRICSAVINGUPDATER_H_*/
