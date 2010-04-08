/*
  <newmo.h>
  Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

  Sébastien Verel, Arnaud Liefooghe, Jérémie Humeau

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

#ifndef _newmo_h
#define _newmo_h

#include <algo/moLocalSearch.h>

#include <comparator/moComparator.h>
#include <comparator/moNeighborComparator.h>
#include <comparator/moSolNeighborComparator.h>
#include <comparator/moSolComparator.h>

#include <continuator/moCheckpoint.h>
#include <continuator/moContinuator.h>
#include <continuator/moCounterMonitorSaver.h>
#include <continuator/moDistanceStat.h>
#include <continuator/moFitnessStat.h>
#include <continuator/moMaxNeighborStat.h>
#include <continuator/moMinNeighborStat.h>
#include <continuator/moNbInfNeighborStat.h>
#include <continuator/moNbSupNeighborStat.h>
#include <continuator/moNeighborhoodStat.h>
#include <continuator/moNeutralDegreeNeighborStat.h>
#include <continuator/moSecondMomentNeighborStat.h>
#include <continuator/moSizeNeighborStat.h>
#include <continuator/moSolutionStat.h>
#include <continuator/moStat.h>
#include <continuator/moStatBase.h>
#include <continuator/moTrueContinuator.h>
#include <continuator/moIterContinuator.h>

#include <eval/moEval.h>
#include <eval/moFullEvalByCopy.h>
#include <eval/moFullEvalByModif.h>

#include <explorer/moFirstImprExplorer.h>
#include <explorer/moHCneutralExplorer.h>
#include <explorer/moMetropolisHastingExplorer.h>
#include <explorer/moNeighborhoodExplorer.h>
#include <explorer/moRandomNeutralWalkExplorer.h>
#include <explorer/moRandomWalkExplorer.h>
#include <explorer/moSimpleHCexplorer.h>
#include <explorer/moRandomBestHCExplorer.h>
#include <explorer/moTSExplorer.h>
#include <explorer/moILSexplorer.h>
#include <explorer/moSAexplorer.h>

#include <memory/moAspiration.h>
#include <memory/moBestImprAspiration.h>
#include <memory/moDiversification.h>
#include <memory/moDummyMemory.h>
#include <memory/moDummyDiversification.h>
#include <memory/moDummyIntensification.h>
#include <memory/moIntensification.h>
#include <memory/moMemory.h>
#include <memory/moSolVectorTabuList.h>
#include <memory/moTabuList.h>
#include <memory/moCountMoveMemory.h>

#include <neighborhood/moBackableNeighbor.h>
#include <neighborhood/moIndexNeighbor.h>
#include <neighborhood/moIndexNeighborhood.h>
#include <neighborhood/moNeighbor.h>
#include <neighborhood/moNeighborhood.h>
#include <neighborhood/moOrderNeighborhood.h>
#include <neighborhood/moRndNeighborhood.h>
#include <neighborhood/moRndWithoutReplNeighborhood.h>
#include <neighborhood/moRndWithReplNeighborhood.h>

#include <perturb/moPerturbation.h>
#include <perturb/moMonOpPerturb.h>
#include <perturb/moRestartPerturb.h>
#include <perturb/moNeighborhoodPerturb.h>

#include <acceptCrit/moAcceptanceCriterion.h>
#include <acceptCrit/moAlwaysAcceptCrit.h>
#include <acceptCrit/moBetterAcceptCrit.h>

#include <coolingSchedule/moCoolingSchedule.h>
#include <coolingSchedule/moSimpleCoolingSchedule.h>

#include <problems/bitString/moBitNeighbor.h>

//#include <old/moMove.h>
//#include <old/moMoveIncrEval.h>
//#include <old/moMoveInit.h>
//#include <old/moNextMove.h>
//#include <old/moMoveNeighbor.h>
//#include <old/moMoveNeighborhood.h>

#endif
