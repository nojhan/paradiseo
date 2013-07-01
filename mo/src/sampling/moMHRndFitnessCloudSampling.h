/*
  <moMHRndFitnessCloudSampling.h>
  Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

  Sebastien Verel, Arnaud Liefooghe, Jeremie Humeau

  This software is governed by the CeCILL license under French law and
  abiding by the rules of distribution of free software.  You can  use,
  modify and/ or redistribute the software under the terms of the CeCILL
  license as circulated by CEA, CNRS and INRIA at the following URL
  "http://www.cecill.info".

  As a counterpart to the access to the source code and  rights to copy,
  modify and redistribute granted by the license, users are provided only
  with a limited warranty  and the software's author,  the holder of the
  economic rights,  and the successive licensors  have only  limited liability.

  In this respect, the user's attention is drawn to the risks associated
  with loading,  using,  modifying and/or developing or reproducing the
  software by the user in light of its specific status of free software,
  that may mean  that it is complicated to manipulate,  and  that  also
  therefore means  that it is reserved for developers  and  experienced
  professionals having in-depth computer knowledge. Users are therefore
  encouraged to load and test the software's suitability as regards their
  requirements in conditions enabling the security of their systems and/or
  data to be ensured and,  more generally, to use and operate it in the
  same conditions as regards security.
  The fact that you are presently reading this means that you have had
  knowledge of the CeCILL license and that you accept its terms.

  ParadisEO WebSite : http://paradiseo.gforge.inria.fr
  Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef moMHRndFitnessCloudSampling_h
#define moMHRndFitnessCloudSampling_h

#include <eoInit.h>
#include <neighborhood/moNeighborhood.h>
#include <eval/moEval.h>
#include <eoEvalFunc.h>
#include <algo/moMetropolisHastings.h>
#include <continuator/moNeighborFitnessStat.h>
#include <sampling/moFitnessCloudSampling.h>

/**
 * To compute an estimation of the fitness cloud,
 *   i.e. the scatter plot of solution fitness versus neighbor fitness:
 *
 *   Here solution are sampled with Metropolis-Hasting method
 *
 *   Sample the fitness of solutions from Metropolis-Hasting sampling
 *   and the fitness of one random neighbor
 *
 *   The values are collected during the Metropolis-Hasting walk
 *
 */
template <class Neighbor>
class moMHRndFitnessCloudSampling : public moFitnessCloudSampling<Neighbor>
{
public:
    typedef typename Neighbor::EOT EOT ;

    using moSampling<Neighbor>::localSearch;
    using moSampling<Neighbor>::checkpoint;
    using moSampling<Neighbor>::monitorVec;
    using moSampling<Neighbor>::continuator;
    using moFitnessCloudSampling<Neighbor>::fitnessStat;

    /**
     * Constructor
     * @param _init initialisation method of the solution
     * @param _neighborhood neighborhood to get one random neighbor (supposed to be random neighborhood)
     * @param _fullEval Fitness function, full evaluation function
     * @param _eval neighbor evaluation, incremental evaluation function
     * @param _nbStep Number of step of the MH sampling
     */
    moMHRndFitnessCloudSampling(eoInit<EOT> & _init,
                                moNeighborhood<Neighbor> & _neighborhood,
                                eoEvalFunc<EOT>& _fullEval,
                                moEval<Neighbor>& _eval,
                                unsigned int _nbStep) :
            moFitnessCloudSampling<Neighbor>(_init, _neighborhood, _fullEval, _eval, _nbStep),
            neighborFitnessStat(_neighborhood, _eval)
    {
        // delete the dummy local search
        delete localSearch;

        // Metropolis-Hasting sampling
        localSearch = new moMetropolisHastings<Neighbor>(_neighborhood, _fullEval, _eval, _nbStep);

        // delete the checkpoint with the wrong continuator
        delete checkpoint;

        // set the continuator
        continuator = localSearch->getContinuator();

        // re-construction of the checkpoint
        checkpoint = new moCheckpoint<Neighbor>(*continuator);
        checkpoint->add(fitnessStat);
        checkpoint->add(*monitorVec[0]);

        // one random neighbor
        this->add(neighborFitnessStat);
    }

protected:
    moNeighborFitnessStat< Neighbor > neighborFitnessStat;

};


#endif
