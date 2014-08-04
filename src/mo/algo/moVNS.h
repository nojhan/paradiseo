/*
  <moVNS.h>
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

#ifndef _moVNS_h
#define _moVNS_h

#include "moLocalSearch.h"
#include "../../eo/eoOp.h"
#include "../comparator/moSolComparator.h"
#include "../continuator/moContinuator.h"

#include "../explorer/moVNSexplorer.h"
#include "../neighborhood/moVariableNeighborhoodSelection.h"
#include "../acceptCrit/moAcceptanceCriterion.h"


/**
 * the "Variable Neighborhood Search" metaheuristic
 */
template<class Neighbor>
class moVNS: public moLocalSearch< Neighbor >
{
public:
  typedef typename Neighbor::EOT EOT;
  typedef moNeighborhood<Neighbor> Neighborhood ;


  /**
   * full constructor for a VNS
   * @param _selection selection the "neighborhood" search heuristics during the search
   * @param _acceptCrit acceptance criteria which compare and accept or not the two solutions
   * @param _fullEval the full evaluation function
   * @param _cont an external continuator
   */
  moVNS(moVariableNeighborhoodSelection<EOT> & _selection,
	moAcceptanceCriterion<Neighbor>& _acceptCrit,
	eoEvalFunc<EOT>& _fullEval,
	moContinuator<Neighbor>& _cont) :
    moLocalSearch<Neighbor>(explorer, _cont, _fullEval),
    explorer(_selection, _acceptCrit)
  {}

private:
  // the explorer of the VNS
  moVNSexplorer<Neighbor> explorer;

};

#endif
