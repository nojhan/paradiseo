/*
* <FlowShopEval.h>
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

#ifndef FLOWSHOPEVAL_H_
#define FLOWSHOPEVAL_H_

#include <vector>
#include <paradiseo/moeo/core/moeoEvalFunc.h>
#include "FlowShop.h"

/**
 * Evaluation of the objective vector a (multi-objective) FlowShop object
 */
class FlowShopEval : public moeoEvalFunc<FlowShop>
  {
  public:

    /**
     * Ctor
     * @param _M the number of machines 
     * @param _N the number of jobs to schedule
     * @param _p the processing times
     * @param _d the due dates
     */
    FlowShopEval(unsigned int _M, unsigned int _N, const std::vector< std::vector<unsigned int> > & _p, const std::vector<unsigned int> & _d);


    /**
     * computation of the multi-objective evaluation of a FlowShop object
     * @param _flowshop the FlowShop object to evaluate
     */
    void operator()(FlowShop & _flowshop);


  private:

    /** number of machines */
    unsigned int M;
    /** number of jobs */
    unsigned int N;
    /** p[i][j] = processing time of job j on machine i */
    std::vector< std::vector < unsigned int > > p;
    /** d[j] = due-date of the job j */
    std::vector < unsigned int > d;


    /**
     * computation of the makespan
     * @param _flowshop the genotype to evaluate
     */
    double makespan(const FlowShop & _flowshop);


    /**
     * computation of the tardiness
     * @param _flowshop the genotype to evaluate
     */
    double tardiness(const FlowShop & _flowshop);


    /**
     * computation of the completion times of a scheduling (for each job on each machine)
     * C[i][j] = completion of the jth job of the scheduling on the ith machine
     * @param _flowshop the genotype to evaluate
     */
    std::vector< std::vector<unsigned int> > completionTime (const FlowShop & _flowshop);

  };

#endif /*FLOWSHOPEVAL_H_*/
