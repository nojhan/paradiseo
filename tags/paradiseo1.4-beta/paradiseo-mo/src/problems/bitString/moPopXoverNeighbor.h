/*
  <moPopXoverNeighbor.h>
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

#ifndef _moPopXoverNeighbor_h
#define _moPopXoverNeighbor_h

#include <eoPop.h>
#include <problems/bitString/moPopSol.h>
#include <neighborhood/moNeighbor.h>

/**
 * Crossover neighbor: apply one crossover operator between 2 random solutions of the population 
 */
template< class EOT >
class moPopXoverNeighbor : public moNeighbor<EOT>
{
public:
  typedef typename EOT::SUBEOT SUBEOT ;

  using moNeighbor<EOT>::fitness;

  /**
   * Move the solution according to the information of this neighbor
   * @param _solution the solution to move
   */
  virtual void move(EOT & _solution) {
    if (_solution.size() > 0) {
      _solution[i1] = sol1;
      _solution[i2] = sol2;

      _solution.invalidate();
    }
  }

  /**
   * Set the index of the solutions on which the crossover is applied
   *
   * @param _i1 index of the first solution
   * @param _i2 index of the second solution
   */
  void setIndexes(unsigned int _i1, unsigned int _i2){
    i1 = _i1;
    i2 = _i2;
  }

  /**
   * Give the variable on the first solution 
   *
   * @return first solution
   */
  SUBEOT& solution1() {
    return sol1;
  }
 
  /**
   * Give the variable on the second solution 
   *
   * @return second solution
   */
  SUBEOT& solution2() {
    return sol2;
  }
 
  /**
   * Give the index in the population of the first solution 
   *
   * @return index of the first solution
   */
  unsigned int index1() {
    return i1;
  }
 
  /**
   * Give the index in the population of the second solution 
   *
   * @return index of the second solution
   */
  unsigned int index2() {
    return i2;
  }
 
  /**
   * return the class name
   * @return the class name as a std::string
   */
  virtual std::string className() const {
    return "moPopXoverNeighbor";
  }

  /**
   * Read object.\
   * Calls base class, just in case that one had something to do.
   * The read and print methods should be compatible and have the same format.
   * In principle, format is "plain": they just print a number
   * @param _is a std::istream.
   * @throw runtime_std::exception If a valid object can't be read.
   */
  virtual void readFrom(std::istream& _is) {
    std::string fitness_str;
    int pos = _is.tellg();
    _is >> fitness_str;
    if (fitness_str == "INVALID") {
      throw std::runtime_error("invalid fitness");
    }
    else {
      typename EOT::Fitness repFit ;
      _is.seekg(pos);
      _is >> repFit;

      _is >> i1;
      _is >> i2;

      _is >> sol1;
      _is >> sol2;

      fitness(repFit);
    }
  }

  /**
   * Write object. Called printOn since it prints the object _on_ a stream.
   * @param _os A std::ostream.
   */
  virtual void printOn(std::ostream& _os) const {
    _os << fitness() << ' ' << i1 << ' ' << i2 << ' ' << sol1 << ' ' << sol2 << std::endl;
  }

private:
  // the two solutions which are the results of the crossover operator
  SUBEOT sol1, sol2;

  // index of the two solutions
  unsigned int i1, i2;
};

#endif
