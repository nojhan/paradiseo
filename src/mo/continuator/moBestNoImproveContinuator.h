/*
  <moBestNoImproveContinuator.h>
  Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

  Sebastien Verel, Arnaud Liefooghe, Jeremie Humeau

  This software is governed by the CeCILL license under French law and
  abiding by the rules of distribution of free software.  You can  ue,
  modify and/ or redistribute the software under the terms of the CeCILL
  license as circulated by CEA, CNRS and INRIA at the following URL
  "http://www.cecill.info".

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

#ifndef _moBestNoImproveContinuator_h
#define _moBestNoImproveContinuator_h

#include "moContinuator.h"
#include "../neighborhood/moNeighborhood.h"
#include "../comparator/moSolComparator.h"

/**
 * Stop when the best solution cannot be improved 
 * within a given number of iterations
 */
template< class Neighbor >
class moBestNoImproveContinuator : public moContinuator<Neighbor>
{
public:
  typedef typename Neighbor::EOT EOT ;

  /**
   * Constructor 
   *
   * @param _bestSol the best solution
   * @param _maxNoImprove number maximum of iterations
   * @param _solComparator a comparator between solutions
   * @param _verbose true/false : verbose mode on/off
   */
  moBestNoImproveContinuator(const EOT & _bestSol, 
			     unsigned int _maxNoImprove, 
			     moSolComparator<EOT>& _solComparator,
			     bool _verbose = true): bestSol(_bestSol), maxNoImprove(_maxNoImprove), solComparator(_solComparator), verbose(_verbose) {}

  /**
   * Constructor where the comparator of solutions is the default comparator
   *
   * @param _bestSol the best solution
   * @param _maxNoImprove number maximum of iterations
   * @param _verbose true/false : verbose mode on/off
   */
  moBestNoImproveContinuator(const EOT & _bestSol, 
			     unsigned int _maxNoImprove, 
			     bool _verbose = true): bestSol(_bestSol), maxNoImprove(_maxNoImprove), solComparator(defaultSolComp), verbose(_verbose) {}

  /**
   * Count and test the number of non improvement of the best solution
   * improvement: if the current solution is STRICTLY better than the current best solution  
   *
   *@param _solution a solution
   *@return true if counter < maxNoImprove
   */
  virtual bool operator()(EOT & _solution) {
    if (solComparator(_solution, bestSol) || solComparator.equals(_solution, bestSol))
      cpt++;
    else
      cpt = 0;

    bool res = (cpt < maxNoImprove);

    if (!res && verbose)
      std::cout << "STOP in moBestNoImproveContinuator: Reached maximum number of iterations without improvement [" << cpt << "/" << maxNoImprove << "]" << std::endl;

    return res;
  }

  /**
   * reset the counter of iteration
   * @param _solution a solution
   */
  virtual void init(EOT & _solution) {
    cpt = 0;
  }

  /**
   * the current number of iteration without improvement
   * @return the number of iteration
   */
  unsigned int value() {
    return cpt ;
  }

private:
  const EOT & bestSol;
  unsigned int maxNoImprove;
  unsigned int cpt;
  // comparator between solutions
  moSolComparator<EOT>& solComparator;
  bool verbose;
  moSolComparator<EOT> defaultSolComp;

};
#endif
