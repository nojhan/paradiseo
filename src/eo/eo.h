//-----------------------------------------------------------------------------
// eo
// (c) GeNeura Team 1998 - 2000
/*
    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Contact: todos@geneura.ugr.es, http://geneura.ugr.es
 */
//-----------------------------------------------------------------------------

#ifdef _MSC_VER
// to avoid long name warnings
#pragma warning(disable:4786)
#endif

#ifndef _eo_
#define _eo_

#ifdef HAVE_CONFIG_H
#include "eo/config.h"
#endif

// general purpose
#include "eo/utils/eoData.h"
#include "eo/eoObject.h"
#include "eo/eoPrintable.h"
#include "eo/eoPersistent.h"
#include "eo/eoScalarFitness.h"
#include "eo/eoDualFitness.h"
#include "eo/EO.h"

#include "eo/utils/rnd_generators.h"
#include "eo/eoFunctor.h"
#include "eo/apply.h"

// eo's
#include "eo/eoVector.h"

#include "eo/other/eoString.h"

#include "eo/utils/eoRndGenerators.h"
#include "eo/eoInit.h"
#include "eo/utils/eoUniformInit.h"

// the variation operators
#include "eo/eoOp.h"
#include "eo/eoGenOp.h"
#include "eo/eoCloneOps.h"
#include "eo/eoOpContainer.h"
// combinations of simple eoOps (eoMonOp and eoQuadOp)
#include "eo/eoProportionalCombinedOp.h"
// didactic (mimics SGA-like variation into an eoGenOp)
// calls crossover and mutation sequentially,
// with their respective mutation rates
#include "eo/eoSGAGenOp.h"
// its dual: crossover, mutation (and copy) - proportional choice
// w.r.t. given relative weights
#include "eo/eoPropGAGenOp.h"

// population
#include "eo/eoPop.h"

// Evaluation functions (all include eoEvalFunc.h)
#include "eo/eoPopEvalFunc.h"
#include "eo/eoEvalFuncPtr.h"
#include "eo/eoEvalCounterThrowException.h"
#include "eo/eoEvalTimeThrowException.h"
#include "eo/eoEvalUserTimeThrowException.h"

// Continuators - all include eoContinue.h
#include "eo/eoCombinedContinue.h"
#include "eo/eoGenContinue.h"
#include "eo/eoSteadyFitContinue.h"
#include "eo/eoEvalContinue.h"
#include "eo/eoFitContinue.h"
#include "eo/eoPeriodicContinue.h"
#include "eo/eoTimeContinue.h" // added th T.Legrand
#ifndef _MSC_VER
#include "eo/eoCtrlCContinue.h"  // CtrlC handling (using 2 global variables!)
#endif
// Selection
// the eoSelectOne's
#include "eo/eoRandomSelect.h"
#include "eo/eoSequentialSelect.h"
#include "eo/eoDetTournamentSelect.h"
#include "eo/eoProportionalSelect.h"
#include "eo/eoFitnessScalingSelect.h" // also contains eoLinearFitScaling.h
#include "eo/eoRankingSelect.h"
#include "eo/eoStochTournamentSelect.h"
#include "eo/eoSharingSelect.h"
// Embedding truncation selection
#include "eo/eoTruncatedSelectOne.h"

// the batch selection - from an eoSelectOne
#include "eo/eoSelectPerc.h"
#include "eo/eoSelectNumber.h"
#include "eo/eoSelectMany.h"
#include "eo/eoTruncatedSelectMany.h"

// other batch selections
// DetSelect can also be obtained as eoSequentialSelect, an eoSelectOne
// (using setup and an index)
#include "eo/eoDetSelect.h"
#include "eo/eoRankMuSelect.h"

// Breeders
#include "eo/eoGeneralBreeder.h"	// applies one eoGenOp, stop on offspring count
// #include "eo/eoOneToOneBreeder.h"	// parent + SINGLE offspring compete (e.g. DE) - not ready yet...

// Replacement
// #include "eo/eoReplacement.h"
#include "eo/eoMergeReduce.h"
#include "eo/eoReduceMerge.h"
#include "eo/eoSurviveAndDie.h"

// a simple transformer
#include "eo/eoSGATransform.h"

// Perf2Worth stuff - includes eoSelectFromWorth.h
#include "eo/eoNDSorting.h"


// Algorithms
#include "eo/eoEasyEA.h"
#include "eo/eoSGA.h"
// #include "eo/eoEvolutionStrategy.h"   removed for a while - until eoGenOp is done

// Utils
#include "eo/utils/checkpointing"
#include "eo/utils/eoRealVectorBounds.h" // includes eoRealBounds.h
#include "eo/utils/eoIntBounds.h"        // no eoIntVectorBounds

// aliens
#include "eo/other/external_eo"
#include "eo/eoCounter.h"


//-----------------------------------------------------------------------------
// to be continued ...
//-----------------------------------------------------------------------------

/*** Particle Swarm Optimization stuff ***/

// basic particle definitions
#include "eo/PO.h"
#include "eo/eoVectorParticle.h"
#include "eo/eoBitParticle.h"
#include "eo/eoRealParticle.h"

// initialization
#include "eo/eoParticleBestInit.h"
#include "eo/eoInitializer.h"

// velocities
#include "eo/eoVelocity.h"
#include "eo/eoStandardVelocity.h"
#include "eo/eoExtendedVelocity.h"
#include "eo/eoIntegerVelocity.h"
#include "eo/eoConstrictedVelocity.h"
#include "eo/eoFixedInertiaWeightedVelocity.h"
#include "eo/eoVariableInertiaWeightedVelocity.h"
#include "eo/eoConstrictedVariableWeightVelocity.h"

// flights
#include "eo/eoFlight.h"
#include "eo/eoStandardFlight.h"
#include "eo/eoVelocityInit.h"
#include "eo/eoBinaryFlight.h"
#include "eo/eoSigBinaryFlight.h"

// topologies
#include "eo/eoTopology.h"
#include "eo/eoStarTopology.h"
#include "eo/eoLinearTopology.h"
#include "eo/eoRingTopology.h"
#include "eo/eoNeighborhood.h"
#include "eo/eoSocialNeighborhood.h"

// PS algorithms
#include "eo/eoPSO.h"
#include "eo/eoEasyPSO.h"
#include "eo/eoSyncEasyPSO.h"

// utils
#include "eo/eoRealBoundModifier.h"
#include "eo/eoRandomRealWeightUp.h"
#include "eo/eoWeightUpdater.h"
#include "eo/eoLinearDecreasingWeightUp.h"
#include "eo/eoGaussRealWeightUp.h"

#include "eo/utils/eoLogger.h"
#include "eo/utils/eoParallel.h"

// ga's
#include "eo/ga/eoBitOp.h"

// es's
#include "eo/es/CMAState.h"
#include "eo/es/CMAParams.h"
#include "eo/es/eoCMAInit.h"
#include "eo/es/eoCMABreed.h"
#include "eo/es/make_es.h"

#endif

// Local Variables:
// mode: C++
// End:
