/* 
* <moeo>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Arnaud Liefooghe
*
* This software is governed by the CeCILL license under French law and
* abiding by the rules of distribution of free software.  You can  use,
* modify and/ or redistribute the software under the terms of the CeCILL
* license as circulated by CEA, CNRS and INRIA at the following URL
* "http://www.cecill.info".
*
* As a counterpart to the access to the source code and  rights to copy,
* modify and redistribute granted by the license, users are provided only
* with a limited warranty  and the software's author,  the holder of the
* economic rights,  and the successive licensors  have only  limited liability.
*
* In this respect, the user's attention is drawn to the risks associated
* with loading,  using,  modifying and/or developing or reproducing the
* software by the user in light of its specific status of free software,
* that may mean  that it is complicated to manipulate,  and  that  also
* therefore means  that it is reserved for developers  and  experienced
* professionals having in-depth computer knowledge. Users are therefore
* encouraged to load and test the software's suitability as regards their
* requirements in conditions enabling the security of their systems and/or
* data to be ensured and,  more generally, to use and operate it in the
* same conditions as regards security.
* The fact that you are presently reading this means that you have had
* knowledge of the CeCILL license and that you accept its terms.
*
* ParadisEO WebSite : http://paradiseo.gforge.inria.fr
* Contact: paradiseo-help@lists.gforge.inria.fr
*
*/
//-----------------------------------------------------------------------------

#ifndef MOEO_
#define MOEO_


#include <paradiseo/eo.h>

#include <paradiseo/moeo/algo/moeoAlgo.h>
#include <paradiseo/moeo/algo/moeoASEEA.h>
#include <paradiseo/moeo/algo/moeoEA.h>
#include <paradiseo/moeo/algo/moeoEasyEA.h>
#include <paradiseo/moeo/algo/moeoIBEA.h>
#include <paradiseo/moeo/algo/moeoMOGA.h>
#include <paradiseo/moeo/algo/moeoNSGA.h>
#include <paradiseo/moeo/algo/moeoNSGAII.h>
#include <paradiseo/moeo/algo/moeoPLS1.h>
#include <paradiseo/moeo/algo/moeoPLS2.h>
#include <paradiseo/moeo/algo/moeoPopAlgo.h>
#include <paradiseo/moeo/algo/moeoPopLS.h>
#include <paradiseo/moeo/algo/moeoSEEA.h>
#include <paradiseo/moeo/algo/moeoSPEA2.h>
#include <paradiseo/moeo/algo/moeoUnifiedDominanceBasedLS.h>

#include <paradiseo/moeo/archive/moeo2DMinHypervolumeArchive.h>
#include <paradiseo/moeo/archive/moeoArchive.h>
#include <paradiseo/moeo/archive/moeoBoundedArchive.h>
#include <paradiseo/moeo/archive/moeoEpsilonHyperboxArchive.h>
#include <paradiseo/moeo/archive/moeoFitDivBoundedArchive.h>
#include <paradiseo/moeo/archive/moeoFixedSizeArchive.h>
#include <paradiseo/moeo/archive/moeoImprOnlyBoundedArchive.h>
#include <paradiseo/moeo/archive/moeoSPEA2Archive.h>
#include <paradiseo/moeo/archive/moeoUnboundedArchive.h>

#include <paradiseo/moeo/comparator/moeoAggregativeComparator.h>
#include <paradiseo/moeo/comparator/moeoComparator.h>
#include <paradiseo/moeo/comparator/moeoDiversityThenFitnessComparator.h>
#include <paradiseo/moeo/comparator/moeoEpsilonObjectiveVectorComparator.h>
#include <paradiseo/moeo/comparator/moeoFitnessComparator.h>
#include <paradiseo/moeo/comparator/moeoFitnessThenDiversityComparator.h>
#include <paradiseo/moeo/comparator/moeoGDominanceObjectiveVectorComparator.h>
#include <paradiseo/moeo/comparator/moeoObjectiveObjectiveVectorComparator.h>
#include <paradiseo/moeo/comparator/moeoObjectiveVectorComparator.h>
#include <paradiseo/moeo/comparator/moeoOneObjectiveComparator.h>
#include <paradiseo/moeo/comparator/moeoParetoObjectiveVectorComparator.h>
#include <paradiseo/moeo/comparator/moeoParetoDualObjectiveVectorComparator.h>
#include <paradiseo/moeo/comparator/moeoPtrComparator.h>
#include <paradiseo/moeo/comparator/moeoStrictObjectiveVectorComparator.h>
#include <paradiseo/moeo/comparator/moeoWeakObjectiveVectorComparator.h>

#include <paradiseo/moeo/core/MOEO.h>
#include <paradiseo/moeo/core/moeoBitVector.h>
#include <paradiseo/moeo/core/moeoEvalFunc.h>
#include <paradiseo/moeo/core/moeoIntVector.h>
#include <paradiseo/moeo/core/moeoObjectiveVector.h>
#include <paradiseo/moeo/core/moeoObjectiveVectorTraits.h>
#include <paradiseo/moeo/core/moeoScalarObjectiveVector.h>
#include <paradiseo/moeo/core/moeoRealObjectiveVector.h>
#include <paradiseo/moeo/core/moeoDualRealObjectiveVector.h>
#include <paradiseo/moeo/core/moeoRealVector.h>
#include <paradiseo/moeo/core/moeoVector.h>

#include <paradiseo/moeo/distance/moeoDistance.h>
#include <paradiseo/moeo/distance/moeoDistanceMatrix.h>
#include <paradiseo/moeo/distance/moeoEuclideanDistance.h>
#include <paradiseo/moeo/distance/moeoManhattanDistance.h>
#include <paradiseo/moeo/distance/moeoNormalizedDistance.h>
#include <paradiseo/moeo/distance/moeoObjSpaceDistance.h>

#include <paradiseo/moeo/diversity/moeoCrowdingDiversityAssignment.h>
#include <paradiseo/moeo/diversity/moeoDiversityAssignment.h>
#include <paradiseo/moeo/diversity/moeoDummyDiversityAssignment.h>
#include <paradiseo/moeo/diversity/moeoFrontByFrontCrowdingDiversityAssignment.h>
#include <paradiseo/moeo/diversity/moeoFrontByFrontSharingDiversityAssignment.h>
#include <paradiseo/moeo/diversity/moeoNearestNeighborDiversityAssignment.h>
#include <paradiseo/moeo/diversity/moeoSharingDiversityAssignment.h>

#include <paradiseo/moeo/explorer/moeoExhaustiveNeighborhoodExplorer.h>
#include <paradiseo/moeo/explorer/moeoFirstImprovingNeighborhoodExplorer.h>
#include <paradiseo/moeo/explorer/moeoNoDesimprovingNeighborhoodExplorer.h>
#include <paradiseo/moeo/explorer/moeoPopNeighborhoodExplorer.h>
#include <paradiseo/moeo/explorer/moeoSimpleSubNeighborhoodExplorer.h>
#include <paradiseo/moeo/explorer/moeoSubNeighborhoodExplorer.h>

#include <paradiseo/moeo/fitness/moeoAggregationFitnessAssignment.h>
#include <paradiseo/moeo/fitness/moeoBinaryIndicatorBasedFitnessAssignment.h>
#include <paradiseo/moeo/fitness/moeoConstraintFitnessAssignment.h>
#include <paradiseo/moeo/fitness/moeoCriterionBasedFitnessAssignment.h>
#include <paradiseo/moeo/fitness/moeoDominanceBasedFitnessAssignment.h>
#include <paradiseo/moeo/fitness/moeoDominanceCountFitnessAssignment.h>
#include <paradiseo/moeo/fitness/moeoDominanceCountRankingFitnessAssignment.h>
#include <paradiseo/moeo/fitness/moeoDominanceDepthFitnessAssignment.h>
#include <paradiseo/moeo/fitness/moeoDominanceRankFitnessAssignment.h>
#include <paradiseo/moeo/fitness/moeoDummyFitnessAssignment.h>
#include <paradiseo/moeo/fitness/moeoExpBinaryIndicatorBasedFitnessAssignment.h>
#include <paradiseo/moeo/fitness/moeoExpBinaryIndicatorBasedDualFitnessAssignment.h>
#include <paradiseo/moeo/fitness/moeoFitnessAssignment.h>
#include <paradiseo/moeo/fitness/moeoIndicatorBasedFitnessAssignment.h>
#include <paradiseo/moeo/fitness/moeoReferencePointIndicatorBasedFitnessAssignment.h>
#include <paradiseo/moeo/fitness/moeoScalarFitnessAssignment.h>
#include <paradiseo/moeo/fitness/moeoSingleObjectivization.h>
#include <paradiseo/moeo/fitness/moeoUnaryIndicatorBasedFitnessAssignment.h>

#include <paradiseo/moeo/hybridization/moeoDMLSGenUpdater.h>
#include <paradiseo/moeo/hybridization/moeoDMLSMonOp.h>

#include <paradiseo/moeo/metric/moeoAdditiveEpsilonBinaryMetric.h>
#include <paradiseo/moeo/metric/moeoContributionMetric.h>
#include <paradiseo/moeo/metric/moeoDistanceMetric.h>
#include <paradiseo/moeo/metric/moeoEntropyMetric.h>
#include <paradiseo/moeo/metric/moeoHypervolumeBinaryMetric.h>
#include <paradiseo/moeo/metric/moeoHyperVolumeDifferenceMetric.h>
#include <paradiseo/moeo/metric/moeoDualHyperVolumeDifferenceMetric.h>
#include <paradiseo/moeo/metric/moeoHyperVolumeMetric.h>
#include <paradiseo/moeo/metric/moeoMetric.h>
#include <paradiseo/moeo/metric/moeoNormalizedSolutionVsSolutionBinaryMetric.h>
#include <paradiseo/moeo/metric/moeoVecVsVecAdditiveEpsilonBinaryMetric.h>
#include <paradiseo/moeo/metric/moeoVecVsVecEpsilonBinaryMetric.h>
#include <paradiseo/moeo/metric/moeoVecVsVecMultiplicativeEpsilonBinaryMetric.h>

#include <paradiseo/moeo/replacement/moeoElitistReplacement.h>
#include <paradiseo/moeo/replacement/moeoEnvironmentalReplacement.h>
#include <paradiseo/moeo/replacement/moeoGenerationalReplacement.h>
#include <paradiseo/moeo/replacement/moeoReplacement.h>

//#include <paradiseo/moeo/scalarStuffs/algo/moeoHC.h>
//#include <paradiseo/moeo/scalarStuffs/algo/moeoILS.h>
//#include <paradiseo/moeo/scalarStuffs/algo/moeoSA.h>
//#include <paradiseo/moeo/scalarStuffs/algo/moeoSolAlgo.h>
//#include <paradiseo/moeo/scalarStuffs/algo/moeoTS.h>
//#include <paradiseo/moeo/scalarStuffs/algo/moeoVFAS.h>
//#include <paradiseo/moeo/scalarStuffs/algo/moeoVNS.h>
#include <paradiseo/moeo/scalarStuffs/archive/moeoArchiveIndex.h>
#include <paradiseo/moeo/scalarStuffs/archive/moeoIndexedArchive.h>
//#include <paradiseo/moeo/scalarStuffs/archive/moeoQuadTree.h>
//#include <paradiseo/moeo/scalarStuffs/archive/moeoQuadTreeArchive.h>
#include <paradiseo/moeo/scalarStuffs/archive/moeoQuickUnboundedArchiveIndex.h>
//#include <paradiseo/moeo/scalarStuffs/distance/moeoAchievementScalarizingFunctionDistance.h>
//#include <paradiseo/moeo/scalarStuffs/distance/moeoAugmentedAchievementScalarizingFunctionDistance.h>
#include <paradiseo/moeo/scalarStuffs/distance/moeoAugmentedWeightedChebychevDistance.h>
//#include <paradiseo/moeo/scalarStuffs/distance/moeoWeightedChebychevDistance.h>
//#include <paradiseo/moeo/scalarStuffs/explorer/moeoHCMoveLoopExpl.h>
//#include <paradiseo/moeo/scalarStuffs/explorer/moeoTSMoveLoopExpl.h>
#include <paradiseo/moeo/scalarStuffs/fitness/moeoAchievementFitnessAssignment.h>
#include <paradiseo/moeo/scalarStuffs/fitness/moeoAchievementScalarizingFunctionMetricFitnessAssignment.h>
#include <paradiseo/moeo/scalarStuffs/fitness/moeoAugmentedAchievementScalarizingFunctionMetricFitnessAssignment.h>
#include <paradiseo/moeo/scalarStuffs/fitness/moeoAugmentedWeightedChebychevMetricFitnessAssignment.h>
//#include <paradiseo/moeo/scalarStuffs/fitness/moeoIncrEvalSingleObjectivizer.h>
//#include <paradiseo/moeo/scalarStuffs/fitness/moeoMetricFitnessAssignment.h>
#include <paradiseo/moeo/scalarStuffs/fitness/moeoWeightedChebychevMetricFitnessAssignment.h>
//#include <paradiseo/moeo/scalarStuffs/weighting/moeoAnytimeWeightStrategy.h>
//#include <paradiseo/moeo/scalarStuffs/weighting/moeoAugmentedQexploreWeightStrategy.h>
//#include <paradiseo/moeo/scalarStuffs/weighting/moeoDichoWeightStrategy.h>
//#include <paradiseo/moeo/scalarStuffs/weighting/moeoDummyRefPointStrategy.h>
//#include <paradiseo/moeo/scalarStuffs/weighting/moeoDummyWeightStrategy.h>
//#include <paradiseo/moeo/scalarStuffs/weighting/moeoFixedTimeBothDirectionWeightStrategy.h>
//#include <paradiseo/moeo/scalarStuffs/weighting/moeoFixedTimeOneDirectionWeightStrategy.h>
//#include <paradiseo/moeo/scalarStuffs/weighting/moeoQexploreWeightStrategy.h>
//#include <paradiseo/moeo/scalarStuffs/weighting/moeoRandWeightStrategy.h>
//#include <paradiseo/moeo/scalarStuffs/weighting/moeoVariableRefPointStrategy.h>
//#include <paradiseo/moeo/scalarStuffs/weighting/moeoVariableWeightStrategy.h>

#include <paradiseo/moeo/selection/moeoDetArchiveSelect.h>
#include <paradiseo/moeo/selection/moeoDetTournamentSelect.h>
#include <paradiseo/moeo/selection/moeoExhaustiveUnvisitedSelect.h>
#include <paradiseo/moeo/selection/moeoNumberUnvisitedSelect.h>
#include <paradiseo/moeo/selection/moeoRandomSelect.h>
#include <paradiseo/moeo/selection/moeoRouletteSelect.h>
#include <paradiseo/moeo/selection/moeoSelectFromPopAndArch.h>
#include <paradiseo/moeo/selection/moeoSelectOne.h>
#include <paradiseo/moeo/selection/moeoSelectors.h>
#include <paradiseo/moeo/selection/moeoStochTournamentSelect.h>
#include <paradiseo/moeo/selection/moeoUnvisitedSelect.h>

#include <paradiseo/moeo/utils/moeoArchiveObjectiveVectorSavingUpdater.h>
#include <paradiseo/moeo/utils/moeoArchiveUpdater.h>
#include <paradiseo/moeo/utils/moeoAverageObjVecStat.h>
#include <paradiseo/moeo/utils/moeoBestObjVecStat.h>
#include <paradiseo/moeo/utils/moeoBinaryMetricSavingUpdater.h>
#include <paradiseo/moeo/utils/moeoBinaryMetricStat.h>
#include <paradiseo/moeo/utils/moeoConvertPopToObjectiveVectors.h>
#include <paradiseo/moeo/utils/moeoDominanceMatrix.h>
#include <paradiseo/moeo/utils/moeoObjectiveVectorNormalizer.h>
#include <paradiseo/moeo/utils/moeoObjVecStat.h>

#include <paradiseo/moeo/continue/moeoHypContinue.h>
#include <paradiseo/moeo/continue/moeoDualHypContinue.h>

#endif /*MOEO_*/
