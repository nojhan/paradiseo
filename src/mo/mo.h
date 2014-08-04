/*
  <mo.h>
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

#include <paradiseo/mo/acceptCrit/moAcceptanceCriterion.h>
#include <paradiseo/mo/acceptCrit/moAlwaysAcceptCrit.h>
#include <paradiseo/mo/acceptCrit/moBetterAcceptCrit.h>

#include <paradiseo/mo/algo/moDummyLS.h>
#include <paradiseo/mo/algo/moFirstImprHC.h>
#include <paradiseo/mo/algo/moILS.h>
#include <paradiseo/mo/algo/moLocalSearch.h>
#include <paradiseo/mo/algo/moMetropolisHasting.h>
#include <paradiseo/mo/algo/moNeutralHC.h>
#include <paradiseo/mo/algo/moRandomBestHC.h>
#include <paradiseo/mo/algo/moRandomNeutralWalk.h>
#include <paradiseo/mo/algo/moRandomSearch.h>
#include <paradiseo/mo/algo/moRandomWalk.h>
#include <paradiseo/mo/algo/moSA.h>
#include <paradiseo/mo/algo/moSimpleHC.h>
#include <paradiseo/mo/algo/moTS.h>
#include <paradiseo/mo/algo/moVNS.h>
#include <paradiseo/mo/algo/eoDummyMonOp.h>

#include <paradiseo/mo/comparator/moComparator.h>
#include <paradiseo/mo/comparator/moNeighborComparator.h>
#include <paradiseo/mo/comparator/moEqualNeighborComparator.h>
#include <paradiseo/mo/comparator/moEqualSolNeighborComparator.h>
#include <paradiseo/mo/comparator/moSolComparator.h>
#include <paradiseo/mo/comparator/moEqualSolComparator.h>
#include <paradiseo/mo/comparator/moSolNeighborComparator.h>

#include <paradiseo/mo/continuator/moAverageFitnessNeighborStat.h>
#include <paradiseo/mo/continuator/moBestSoFarStat.h>
#include <paradiseo/mo/continuator/moBestFitnessStat.h>
#include <paradiseo/mo/continuator/moUnsignedStat.h>
#include <paradiseo/mo/continuator/moValueStat.h>
#include <paradiseo/mo/continuator/moBooleanStat.h>
#include <paradiseo/mo/continuator/moCheckpoint.h>
#include <paradiseo/mo/continuator/moCombinedContinuator.h>
#include <paradiseo/mo/continuator/moContinuator.h>
#include <paradiseo/mo/continuator/moCounterMonitorSaver.h>
#include <paradiseo/mo/continuator/moCounterStat.h>
#include <paradiseo/mo/continuator/moDistanceStat.h>
#include <paradiseo/mo/continuator/moFitContinuator.h>
#include <paradiseo/mo/continuator/moFitnessStat.h>
#include <paradiseo/mo/continuator/moFullEvalContinuator.h>
#include <paradiseo/mo/continuator/moEvalsContinuator.h>
#include <paradiseo/mo/continuator/moIterContinuator.h>
#include <paradiseo/mo/continuator/moMaxNeighborStat.h>
#include <paradiseo/mo/continuator/moMinNeighborStat.h>
#include <paradiseo/mo/continuator/moMinusOneCounterStat.h>
#include <paradiseo/mo/continuator/moNbInfNeighborStat.h>
#include <paradiseo/mo/continuator/moNbSupNeighborStat.h>
#include <paradiseo/mo/continuator/moNeighborBestStat.h>
#include <paradiseo/mo/continuator/moNeighborEvalContinuator.h>
#include <paradiseo/mo/continuator/moNeighborFitnessStat.h>
#include <paradiseo/mo/continuator/moNeighborhoodStat.h>
#include <paradiseo/mo/continuator/moNeutralDegreeNeighborStat.h>
#include <paradiseo/mo/continuator/moSecondMomentNeighborStat.h>
#include <paradiseo/mo/continuator/moSizeNeighborStat.h>
#include <paradiseo/mo/continuator/moSolutionStat.h>
#include <paradiseo/mo/continuator/moStat.h>
#include <paradiseo/mo/continuator/moStatBase.h>
#include <paradiseo/mo/continuator/moStatFromStat.h>
#include <paradiseo/mo/continuator/moStdFitnessNeighborStat.h>
#include <paradiseo/mo/continuator/moTimeContinuator.h>
#include <paradiseo/mo/continuator/moTrueContinuator.h>
#include <paradiseo/mo/continuator/moVectorMonitor.h>

#include <paradiseo/mo/coolingSchedule/moCoolingSchedule.h>
#include <paradiseo/mo/coolingSchedule/moDynSpanCoolingSchedule.h>
#include <paradiseo/mo/coolingSchedule/moSimpleCoolingSchedule.h>
#include <paradiseo/mo/coolingSchedule/moDynSpanCoolingSchedule.h>

#include <paradiseo/mo/eval/moDummyEval.h>
#include <paradiseo/mo/eval/moEval.h>
#include <paradiseo/mo/eval/moEvalCounter.h>
#include <paradiseo/mo/eval/moFullEvalByCopy.h>
#include <paradiseo/mo/eval/moFullEvalByModif.h>
#include <paradiseo/mo/eval/moDoubleIncrNeighborhoodEval.h>

#include <paradiseo/mo/explorer/moDummyExplorer.h>
#include <paradiseo/mo/explorer/moFirstImprHCexplorer.h>
#include <paradiseo/mo/explorer/moILSexplorer.h>
#include <paradiseo/mo/explorer/moMetropolisHastingExplorer.h>
#include <paradiseo/mo/explorer/moNeighborhoodExplorer.h>
#include <paradiseo/mo/explorer/moNeutralHCexplorer.h>
#include <paradiseo/mo/explorer/moRandomBestHCexplorer.h>
#include <paradiseo/mo/explorer/moRandomNeutralWalkExplorer.h>
#include <paradiseo/mo/explorer/moRandomSearchExplorer.h>
#include <paradiseo/mo/explorer/moRandomWalkExplorer.h>
#include <paradiseo/mo/explorer/moSAexplorer.h>
#include <paradiseo/mo/explorer/moSimpleHCexplorer.h>
#include <paradiseo/mo/explorer/moTSexplorer.h>
#include <paradiseo/mo/explorer/moVNSexplorer.h>

#include <paradiseo/mo/memory/moAspiration.h>
#include <paradiseo/mo/memory/moBestImprAspiration.h>
#include <paradiseo/mo/memory/moCountMoveMemory.h>
#include <paradiseo/mo/memory/moDiversification.h>
#include <paradiseo/mo/memory/moDummyDiversification.h>
#include <paradiseo/mo/memory/moDummyIntensification.h>
#include <paradiseo/mo/memory/moDummyMemory.h>
#include <paradiseo/mo/memory/moIndexedVectorTabuList.h>
#include <paradiseo/mo/memory/moIntensification.h>
#include <paradiseo/mo/memory/moMemory.h>
#include <paradiseo/mo/memory/moMonOpDiversification.h>
#include <paradiseo/mo/memory/moNeighborVectorTabuList.h>
#include <paradiseo/mo/memory/moRndIndexedVectorTabuList.h>
#include <paradiseo/mo/memory/moSolVectorTabuList.h>
#include <paradiseo/mo/memory/moRndIndexedVectorTabuList.h>
#include <paradiseo/mo/memory/moTabuList.h>

#include <paradiseo/mo/neighborhood/moBackableNeighbor.h>
#include <paradiseo/mo/neighborhood/moBackwardVectorVNSelection.h>
#include <paradiseo/mo/neighborhood/moDummyNeighbor.h>
#include <paradiseo/mo/neighborhood/moDummyNeighborhood.h>
#include <paradiseo/mo/neighborhood/moForwardVectorVNSelection.h>
#include <paradiseo/mo/neighborhood/moIndexNeighbor.h>
#include <paradiseo/mo/neighborhood/moIndexNeighborhood.h>
#include <paradiseo/mo/neighborhood/moNeighbor.h>
#include <paradiseo/mo/neighborhood/moNeighborhood.h>
#include <paradiseo/mo/neighborhood/moOrderNeighborhood.h>
#include <paradiseo/mo/neighborhood/moRndNeighborhood.h>
#include <paradiseo/mo/neighborhood/moRndVectorVNSelection.h>
#include <paradiseo/mo/neighborhood/moRndWithoutReplNeighborhood.h>
#include <paradiseo/mo/neighborhood/moRndWithReplNeighborhood.h>
#include <paradiseo/mo/neighborhood/moVariableNeighborhoodSelection.h>
#include <paradiseo/mo/neighborhood/moVectorVNSelection.h>
#include <paradiseo/mo/neighborhood/moEvaluatedNeighborhood.h>

#include <paradiseo/mo/perturb/moLocalSearchInit.h>
#include <paradiseo/mo/perturb/moMonOpPerturb.h>
#include <paradiseo/mo/perturb/moNeighborhoodPerturb.h>
#include <paradiseo/mo/perturb/moPerturbation.h>
#include <paradiseo/mo/perturb/moRestartPerturb.h>
#include <paradiseo/mo/perturb/moSolInit.h>

#include <paradiseo/mo/problems/bitString/moBitNeighbor.h>
#include <paradiseo/mo/problems/bitString/moBitsNeighbor.h>
#include <paradiseo/mo/problems/bitString/moBitsNeighborhood.h>
#include <paradiseo/mo/problems/bitString/moBitsWithoutReplNeighborhood.h>
#include <paradiseo/mo/problems/bitString/moBitsWithReplNeighborhood.h>

#include <paradiseo/mo/problems/permutation/moIndexedSwapNeighbor.h>
#include <paradiseo/mo/problems/permutation/moShiftNeighbor.h>
#include <paradiseo/mo/problems/permutation/moSwapNeighbor.h>
#include <paradiseo/mo/problems/permutation/moSwapNeighborhood.h>
#include <paradiseo/mo/problems/permutation/moTwoOptExNeighbor.h>
#include <paradiseo/mo/problems/permutation/moTwoOptExNeighborhood.h>

//#include <paradiseo/mo/problems/eval/moMaxSATincrEval.h>
//#include <paradiseo/mo/problems/eval/moOneMaxIncrEval.h>
//#include <paradiseo/mo/problems/eval/moQAPIncrEval.h>
//#include <paradiseo/mo/problems/eval/moRoyalRoadIncrEval.h>
//#include <paradiseo/mo/problems/eval/moUBQPSimpleIncrEval.h>
//#include <paradiseo/mo/problems/eval/moUBQPdoubleIncrEvaluation.h>
//#include <paradiseo/mo/problems/eval/moUBQPBitsIncrEval.h>


#include <paradiseo/mo/sampling/moAdaptiveWalkSampling.h>
#include <paradiseo/mo/sampling/moAutocorrelationSampling.h>
#include <paradiseo/mo/sampling/moDensityOfStatesSampling.h>
#include <paradiseo/mo/sampling/moFDCsampling.h>
#include <paradiseo/mo/sampling/moFitnessCloudSampling.h>
#include <paradiseo/mo/sampling/moHillClimberSampling.h>
#include <paradiseo/mo/sampling/moAdaptiveWalkSampling.h>
#include <paradiseo/mo/sampling/moMHBestFitnessCloudSampling.h>
#include <paradiseo/mo/sampling/moMHRndFitnessCloudSampling.h>
#include <paradiseo/mo/sampling/moNeutralDegreeSampling.h>
#include <paradiseo/mo/sampling/moNeutralWalkSampling.h>
#include <paradiseo/mo/sampling/moRndBestFitnessCloudSampling.h>
#include <paradiseo/mo/sampling/moRndRndFitnessCloudSampling.h>
#include <paradiseo/mo/sampling/moSampling.h>
#include <paradiseo/mo/sampling/moStatistics.h>

#endif
