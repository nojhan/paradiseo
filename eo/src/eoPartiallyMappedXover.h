/*
  <eoPartiallyMappedXover.h>
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
#ifndef eoPartiallyMappedXover__h
#define eoPartiallyMappedXover__h

//-----------------------------------------------------------------------------

#include <algorithm>
#include <utils/eoRNG.h>

/**
 *
 * Partially Mapped CrossOver (PMX)
 * for permutation representation
 *
 */
template <class EOT> 
class eoPartiallyMappedXover : public eoQuadOp<EOT> 
{
public:
  /**
   *
   * @param _solution1 The first solution
   * @param _solution2 The second solution
   * @return true if the solution has changed
   */
  bool operator()(EOT & _solution1, EOT & _solution2) {
    if (_solution1.size() > 1) {
      // random indexes such that i1 < i2
      int i1 = rng.random(_solution1.size());
      int i2 = rng.random(_solution1.size());

      while (i1 == i2)
	i2 = rng.random(_solution1.size());
      
      if (i1 > i2) {
	int tmp = i1;
	i1 = i2;
	i2 = tmp;
      }
      
      // the permutations between s1 and s2 
      int * p1 = new int[_solution1.size()];
      int * p2 = new int[_solution1.size()];
      
      int i;
      for(i = 0; i < _solution1.size(); i++) {
	p1[i] = -1;
	p2[i] = -1;
      }

      for(i = i1; i <= i2; i++) {
	p1[ _solution2[i] ] = _solution1[i] ;
	p2[ _solution1[i] ] = _solution2[i] ;
      }

      // replace if necessary
      for(i = 0; i < i1; i++) {
	while (p1[ _solution1[i] ] != -1) 
	  _solution1[i] = p1[_solution1[i]];
	while (p2[ _solution2[i] ] != -1) 
	  _solution2[i] = p2[_solution2[i]];
      }      

      // swap between solution1 and solution2 for [i1..i2]
      for(i = i1; i <= i2; i++) {
	_solution1[i] = p2[ _solution1[i] ];
	_solution2[i] = p1[ _solution2[i] ];
      }

      // replace if necessary
      for(i = i2 + 1; i < _solution1.size(); i++) {
	while (p1[ _solution1[i] ] != -1) 
	  _solution1[i] = p1[_solution1[i]];
	while (p2[ _solution2[i] ] != -1) 
	  _solution2[i] = p2[_solution2[i]];
      }      

      // invalidate the solutions because they have been modified
      _solution1.invalidate();
      _solution2.invalidate();

      delete [] p1;
      delete [] p2;

      return true;
    } else
      return false;
  }

  /**
   * The class name.
   */
  virtual std::string className() const { 
    return "eoPartiallyMappedXover"; 
  }

};

#endif
