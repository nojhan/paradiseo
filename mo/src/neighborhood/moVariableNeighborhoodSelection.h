/*
  <moVariableNeighborhoodSelection.h>
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

#ifndef _moVariableNeighborhoodSelection_h
#define _moVariableNeighborhoodSelection_h

#include <paradiseo/eo/eoOp.h>
#include <vector>

/**
 *  This class is used for the Variable Neighborhood Search explorer
 *  It gives the sequence of search heuristics based on the different "neighborhoods" 
 *  The class is built such as the moNeighborhood" with init, next, cont
 *    and two methods to get the heuristics which shake the solution, and which give the local search
 *
 */
template< class EOT >
class moVariableNeighborhoodSelection
{
public:

  /**
   * Return the class id.
   * @return the class name as a std::string
   */
  virtual std::string className() const {
    return "moVariableNeighborhoodSelection";
  }

  /**
   * test if there is still some search heuristics to use
   * @return true if there is some neighborhood to explore
   */
  virtual bool cont(EOT& _solution) = 0;

  /**
   * put on the first search heuristics
   */
  virtual void init(EOT& _solution) = 0;

  /**
   * put the next search heuristics
   */
  virtual void next(EOT& _solution) = 0;
  
  /**
   * Get the current "shake" operator based on the current neighborhood
   *
   * @return current shake operator
   */
  virtual eoMonOp<EOT> & getShake() = 0;

  /**
   * Get the current local search based on the current neighborhood
   *
   * @return current local search 
   */
  virtual eoMonOp<EOT> & getLocalSearch() = 0;

};

#endif
