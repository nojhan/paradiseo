/*
<longKPathEval.h>
Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

Sébastien Verel

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

#ifndef __longKPathEval_h
#define __longKPathEval_h

#include "../../eo/eoEvalFunc.h"

/**
 * Full evaluation function for long k-path problem
 */
template< class EOT >
class LongKPathEval : public eoEvalFunc<EOT>
{
private:
  // parameter k of the problem
  unsigned k;

  // tempory variable if the solution is in the long path
  bool inPath;

  /**
   * compute the number k j in solution = u1^kO^jw between i and i - k + 1 with |u1^kO^j| = i and k+j <= k
   *
   * @param solution the solution to evaluate
   * @param l last position in the bit string
   * @param n0 number of consecutive 0
   * @param n1 number of consecutive 1
   */
  void nbOnesZeros(EOT & solution, unsigned l, unsigned & n0, unsigned & n1) {
    n0 = 0;

    unsigned ind = l - 1;

    while (n0 < k && solution[ind - n0] == 0) 
      n0++;

    n1 = 0;

    ind = ind - n0;
    while (n0 + n1 < k && solution[ind - n1] == 1)
      n1++;
  }

  /**
   * true if the solution is the last solution of the path of bitstring length l
   *
   * @param solution the solution to evaluate
   * @param l size of the path, last position in the bit string
   * @return true if the solution is the solution of the path
   */
  bool final(EOT & solution, unsigned l) {
    if (l == 1)
      return (solution[0] == 1);
    else {
      int i = 0;

      while (i < l - k && solution[i] == 0)
	i++;

      if (i < l - k)
	return false;
      else {
	while (i < l && solution[i] == 1)
	  i++;
	
	return (i == l);
      }
    }
  }

  /**
   * position in the long path
   *
   * @param solution the solution to evaluate
   * @param l size of the path, last position in the bit string
   * @return position in the path
   */
  unsigned rank(EOT & solution, unsigned int l) { 
    if (l == 1) { // long path l = 1
      inPath = true;

      if (solution[0] == 0) 
	return 0;
      else
	return 1;
    } else { // long path for l>1
      unsigned n0, n1;

      // read the k last bits, and count the number of last successive 0 follow by the last successive 1
      nbOnesZeros(solution, l, n0, n1);

      if (n0 == k) // first part of the path
	return rank(solution, l - k);
      else
	if (n1 == k) { // last part of the path
	  return (k+1) * (1 << ((l-1) / k)) - k - rank(solution, l - k);
	} else 
	  if (n0 + n1 == k) {
	    if (final(solution, l - k)) {
	      inPath = true;
	      return (k+1) * (1 << ((l-k-1) / k)) - k + n1;
	    } else {
	      inPath = false;
	      return 0;
	    }
	  } else {
	    inPath = false;
	    return 0;
	  }
    }
  }

  /**
   * compute the number of zero of the bit string
   *
   * @param solution the solution to evaluate
   * @return number of zero in the bit string
   */
  unsigned int nbZero(EOT & solution){
    unsigned int res = 0;

    for(unsigned int i=0; i < solution.size(); i++)
      if (solution[i] == 0)
	res++;

    return res;
  }

public:
  /**
   * Default constructor
   *
   * @param _k parameter k of the long K-path problem
   */
  LongKPathEval(unsigned _k) : k(_k) {
  };

  /**
   * default destructor
   */
  ~LongKPathEval(void) {} ;
    
  /**
   * compute the fitnes of the solution
   *
   * @param solution the solution to evaluate
   * @return fitness of the solution
   */
  void operator()(EOT & solution) {
    inPath = true;

    unsigned r = rank(solution, solution.size());

    if (inPath)
      solution.fitness(solution.size() + r);
    else
      solution.fitness(nbZero(solution));
  }

};

#endif
