/*
  <moVectorVNSelection.h>
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

#ifndef _moVectorVNSelection_h
#define _moVectorVNSelection_h

#include "moVariableNeighborhoodSelection.h"
#include <paradiseo/eo/eoOp.h>

/**
 *  This class is used for the Variable Neighborhood Search explorer inherits from moVariableNeighborhoodSelection
 *  The search heuristics are saved in vectors
 *  The way to croos the vector is not defined here
 *
 */
template< class EOT >
class moVectorVNSelection: public moVariableNeighborhoodSelection<EOT>{

public:

  /**
   * Default constructor with first search heuristics
   *
   * @param _firstLS first local search 
   * @param _firstShake first heuristic which perturbs the solution
   */
  moVectorVNSelection(eoMonOp<EOT>& _firstLS, eoMonOp<EOT>& _firstShake){
    LSvector.push_back(&_firstLS);
    shakeVector.push_back(&_firstShake);

    current = 0;
  }
  
  /**
   * Add some search heuristics
   *
   * @param _otherLS the added local search 
   * @param _otherShake the added heuristic which perturbs the solution
   */
  void add(eoMonOp<EOT>& _otherLS, eoMonOp<EOT>& _otherShake){
    LSvector.push_back(&_otherLS);
    shakeVector.push_back(&_otherShake);
  }

  /**
   * Return the class id.
   * @return the class name as a std::string
   */
  virtual std::string className() const {
    return "moVectorVNSelection";
  }

  /**
   * Get the current "shake" operator based on the current neighborhood
   *
   * @return current shake operator
   */
  virtual eoMonOp<EOT> & getShake() {
    return *(shakeVector[current]);
  }

  /**
   * Get the current local search based on the current neighborhood
   *
   * @return current local search 
   */
  virtual eoMonOp<EOT> & getLocalSearch() {
    return *(LSvector[current]);
  }

protected:
  // vector of local searches
  std::vector<eoMonOp<EOT>* > LSvector;
  // vector of "shake" heiristics which perturbs the current solution
  std::vector<eoMonOp<EOT>* > shakeVector;
  // index of the current search heuristics which is applied
  unsigned int current;

};


#endif
