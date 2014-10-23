/*
  <moNeutralWalkSampling.h>
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

#ifndef moNeutralWalkSampling_h
#define moNeutralWalkSampling_h

#include <eoInit.h>
#include <eval/moEval.h>
#include <eoEvalFunc.h>
#include <algo/moRandomNeutralWalk.h>
#include <sampling/moSampling.h>
#include <perturb/moSolInit.h>
#include <continuator/moSolutionStat.h>
#include <utils/eoDistance.h>
#include <continuator/moDistanceStat.h>
#include <continuator/moNeighborhoodStat.h>
#include <continuator/moMaxNeighborStat.h>
#include <continuator/moMinNeighborStat.h>
#include <continuator/moQ1NeighborStat.h>
#include <continuator/moQ3NeighborStat.h>
#include <continuator/moMedianNeighborStat.h>
#include <continuator/moAverageFitnessNeighborStat.h>
#include <continuator/moStdFitnessNeighborStat.h>
#include <continuator/moSizeNeighborStat.h>
#include <continuator/moNbInfNeighborStat.h>
#include <continuator/moNbSupNeighborStat.h>
#include <continuator/moNeutralDegreeNeighborStat.h>

/**
 * To explore the evolvability of solutions in a neutral networks:
 *   Perform a random neutral walk based on the neighborhood,
 *   The measures of evolvability of solutions are collected during the random neutral walk
 *   The distribution and autocorrelation can be computed from the serie of values
 *
 *    Informations collected:
 *         - the current solution of the walk
 *         - the distance from the starting solution
 *         - the average fitness
 *         - the standard deviation of the fitness
 *         - the minimal fitness in the neighborhood
 *         - the first quartile of fitness in the neighborhood
 *         - the median fitness in the neighborhood
 *         - the third quartile of fitness in the neighborhood
 *         - the maximal fitness
 *         - the size of the neighborhood
 *         - the number of neighbors with lower fitness
 *         - the number of neighbors with equal fitness (neutral degree)
 *         - the number of neighbors with higher fitness
 */
template <class Neighbor>
class moNeutralWalkSampling : public moSampling<Neighbor>
{
public:
  typedef typename Neighbor::EOT EOT ;

  using moSampling<Neighbor>::localSearch;

  /**
   * Constructor
   * @param _initSol the first solution of the walk
   * @param _neighborhood neighborhood giving neighbor in random order
   * @param _fullEval Fitness function, full evaluation function
   * @param _eval neighbor evaluation, incremental evaluation function
   * @param _distance component to measure the distance from the initial solution
   * @param _nbStep Number of steps of the random walk
   */
  moNeutralWalkSampling(EOT & _initSol,
			moNeighborhood<Neighbor> & _neighborhood,
			eoEvalFunc<EOT>& _fullEval,
			moEval<Neighbor>& _eval,
			eoDistance<EOT> & _distance,
			unsigned int _nbStep) :
    moSampling<Neighbor>(init, * new moRandomNeutralWalk<Neighbor>(_neighborhood, _fullEval, _eval, _nbStep), solutionStat),
    init(_initSol),
    distStat(_distance, _initSol),
    neighborhoodStat(_neighborhood, _eval),
    averageStat(neighborhoodStat),
    stdStat(neighborhoodStat),
    minStat(neighborhoodStat),
    maxStat(neighborhoodStat),
    nbSupStat(neighborhoodStat),
    nbInfStat(neighborhoodStat),
    sizeStat(neighborhoodStat),
      ndStat(neighborhoodStat),
      q1Stat(neighborhoodStat),
      medianStat(neighborhoodStat),
      q3Stat(neighborhoodStat)
  {
    this->add(neighborhoodStat, false);
    this->add(distStat);
    this->add(averageStat);
    this->add(stdStat);
    this->add(minStat);
    this->add(q1Stat);
    this->add(medianStat);
    this->add(q3Stat);
    this->add(maxStat);
    this->add(sizeStat);
    this->add(nbInfStat);
    this->add(ndStat);
    this->add(nbSupStat);
  }

  /**
   * Constructor
   * @param _initSol the first solution of the walk
   * @param _neighborhood neighborhood giving neighbor in random order
   * @param _fullEval Fitness function, full evaluation function
   * @param _eval neighbor evaluation, incremental evaluation function
   * @param _distance component to measure the distance from the initial solution
   * @param _nbStep Number of steps of the random walk
   */
  moNeutralWalkSampling(eoInit<EOT> & _init,
			moNeighborhood<Neighbor> & _neighborhood,
			eoEvalFunc<EOT>& _fullEval,
			moEval<Neighbor>& _eval,
			unsigned int _nbStep) :
    moSampling<Neighbor>(_init, * new moRandomNeutralWalk<Neighbor>(_neighborhood, _fullEval, _eval, _nbStep), solutionStat),
    init(initialSol),
    distStat(dummyDistance, initialSol),
    neighborhoodStat(_neighborhood, _eval),
    minStat(neighborhoodStat),
    averageStat(neighborhoodStat),
    stdStat(neighborhoodStat),
    maxStat(neighborhoodStat),
    nbSupStat(neighborhoodStat),
    nbInfStat(neighborhoodStat),
    sizeStat(neighborhoodStat),
      ndStat(neighborhoodStat),
      q1Stat(neighborhoodStat),
      medianStat(neighborhoodStat),
      q3Stat(neighborhoodStat)
  {
    this->add(neighborhoodStat, false);
    //    this->add(distStat);
    this->add(averageStat);
    this->add(stdStat);
    this->add(minStat);
    this->add(q1Stat);
    this->add(medianStat);
    this->add(q3Stat);
    this->add(maxStat);
    this->add(sizeStat);
    this->add(nbInfStat);
    this->add(ndStat);
    this->add(nbSupStat);
  }

  /**
   * default destructor
   */
  ~moNeutralWalkSampling() {
    // delete the pointer on the local search which has been constructed in the constructor
    delete localSearch;
  }

protected:
  EOT initialSol;
  eoHammingDistance<EOT> dummyDistance;
  moSolInit<EOT> init;
  moSolutionStat<EOT> solutionStat;
  moDistanceStat<EOT> distStat;
  moNeighborhoodStat< Neighbor > neighborhoodStat;
  moMinNeighborStat< Neighbor > minStat;
  moAverageFitnessNeighborStat< Neighbor > averageStat;
  moStdFitnessNeighborStat< Neighbor > stdStat;
  moMaxNeighborStat< Neighbor > maxStat;
  moNbSupNeighborStat< Neighbor > nbSupStat;
  moNbInfNeighborStat< Neighbor > nbInfStat;
  moSizeNeighborStat< Neighbor > sizeStat;
  moNeutralDegreeNeighborStat< Neighbor > ndStat;
  moQ1NeighborStat< Neighbor > q1Stat;
  moMedianNeighborStat< Neighbor > medianStat;
  moQ3NeighborStat< Neighbor > q3Stat;

};


#endif
