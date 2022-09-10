/*
  <moBackwardVectorVNSelection.h>
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

#ifndef _moBackwardVectorVNSelection_h
#define _moBackwardVectorVNSelection_h

#include <neighborhood/moVectorVNSelection.h>

/**
 *  This class is used for the Variable Neighborhood Search explorer inherits from moVectorVNSelection
 *  The search heuristics are saved in vectors
 *  They are given in backward order from the last ones to the first ones
 *
 */
template< class EOT >
class moBackwardVectorVNSelection: public moVectorVNSelection<EOT>{

  using moVectorVNSelection<EOT>::LSvector;
  using moVectorVNSelection<EOT>::current;
  
public:
  
  /**
   * Default constructor with first search heuristics
   *
   * @param _firstLS first local search 
   * @param _firstShake first heuristic which perturbs the solution
   * @param _cycle when true, the first heuristics follows the last ones. Otherwise the search stop.
   */
  moBackwardVectorVNSelection(eoMonOp<EOT>& _firstLS, eoMonOp<EOT>& _firstShake, bool _cycle = true) : moVectorVNSelection<EOT>(_firstLS, _firstShake), cycle(_cycle){}
  
  /**
   * Return the class id.
   * @return the class name as a std::string
   */
  virtual std::string className() const {
    return "moBackwardVectorVNSelection";
  }

  /**
   * test if there is still some heuristics 
   *
   * @param _solution the current solution
   * @return true if there is some heuristics
   */
  virtual bool cont(EOT& /*_solution*/){
    return (cycle || (current > 0));
  }

  /**
   * put the current heuristics on the first ones
   *
   * @param _solution the current solution
   */
  virtual void init(EOT& /*_solution*/){
    current = LSvector.size() - 1;
  }

  /**
   * put the current heuristics on the next ones
   *
   * @param _solution the current solution
   */
  virtual void next(EOT& /*_solution*/){
    current = (current + LSvector.size() -1) % LSvector.size();
  }

private:
  // boolean to indicate the last heuristics follow the first ones
  bool cycle;
  
};

#endif
