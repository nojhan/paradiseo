/*
<bbRoyalRoadEval.h>
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

#ifndef _bbRoyalRoadEval_h
#define _bbRoyalRoadEval_h

#include "../../eo/eoEvalFunc.h"
#include <vector>

/**
 * Full evaluation Function for Building-Block Royal Road problem:
 * Richard A. Watson & Thomas Jansen, "A building-block royal road where crossover is provably essential", GECCO 07.
 */
template< class EOT >
class BBRoyalRoadEval : public eoEvalFunc<EOT>
{
public:
  /**
   * Default constructor
   * @param _b number of blocks
   * @param _k size of a block
   */
  BBRoyalRoadEval(unsigned int _b, unsigned int _k) : k(_k), b(_b) {  }

  /**
   * add a target to sub-objective functions
   *
   * @param target target vector of boolean (of size k)
   * @param w weights of this target
   */
  void addTarget(vector<bool> & target, double w) {
    targets.push_back(target);
    weights.push_back(w);
  }
  
  /**
   * Count the number of complete blocks in the bit string
   * @param _sol the solution to evaluate
   */
  void operator() (EOT& _solution) {
    double sum = 0;

    unsigned int i, j, t;
    unsigned int offset;

    // Hamming distance
    double d;

    for(i = 0; i < b; i++) {
      offset = i * k;

      for(t = 0; t < targets.size(); t++) { 
	d = 0;
	for(j = 0; j < k; j++)
	  if (_solution[offset + j] != targets[t][j])
	    d++;

	if (d == 0)
	  sum += weights[t];
	else 
	  sum += 1.0 / ( 1.0 + d ); 
      }
    }

    _solution.fitness(sum);
  }

  /**
   * get the size of a block
   * @return block size
   */
  unsigned int blockSize() {
    return k;
  }

  /**
   * get the number of blocks
   * @return the number of blocks
   */
  unsigned int nbBlocks() {
    return b;
  }

  /**
   * get the targets
   * @return the vector of targets which is a boolean vector
   */
  vector<vector<bool> > & getTargets() {
    return targets;
  }

private:
  // number of blocks
  unsigned int b;

  // size of a block
  unsigned int k;    
    
  vector<vector<bool> > targets;
  vector<double> weights;
};

#endif
