/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

-----------------------------------------------------------------------------
    eoPopEvalFunc.h
    Abstract class for global evaluation of the population

    (c) GeNeura Team, 2000

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

#ifndef eoPopEvalFunc_H
#define eoPopEvalFunc_H

#include "eoEvalFunc.h"
#include "apply.h"

# ifdef WITH_MPI
#include "mpi/eoMpi.h"
#include "mpi/eoTerminateJob.h"
#include "mpi/eoMpiAssignmentAlgorithm.h"
#include "mpi/eoParallelApply.h"
#include "utils/eoParallel.h"

#include <cmath> // ceil
# endif // WITH_MPI

/** eoPopEvalFunc: This abstract class is for GLOBAL evaluators
 *    of a population after variation.
 *    It takes 2 populations (typically the parents and the offspring)
 *    and is suppposed to evaluate them alltogether
 *
 *  Basic use: apply an embedded eoEvalFunc to the offspring
 *
 *  Time-varying fitness: apply the embedded eoEvalFunc to both
 *     offspring and parents
 *
 *  Advanced uses: Co-evolution or "parisian" approach, or ...
 *
 *  Basic parallelization (synchronous standard evolution engine):
 *    call the slaves and wait for the results
 *
 *    @ingroup Evaluation
 */
template<class EOT>
class eoPopEvalFunc : public eoBF<eoPop<EOT> & , eoPop<EOT> &, void>
{};

/////////////////////////////////////////////////////////////
//           eoPopLoopEval
/////////////////////////////////////////////////////////////

/** eoPopLoopEval: an instance of eoPopEvalFunc that simply applies
 *     a private eoEvalFunc to all offspring
 *
 *    @ingroup Evaluation
 */
template<class EOT>
class eoPopLoopEval : public eoPopEvalFunc<EOT> {
public:
  /** Ctor: set value of embedded eoEvalFunc */
  eoPopLoopEval(eoEvalFunc<EOT> & _eval) : eval(_eval) {}

  /** Do the job: simple loop over the offspring */
  void operator()(eoPop<EOT> & _parents, eoPop<EOT> & _offspring)
  {
      (void)_parents;
      apply<EOT>(eval, _offspring);
  }

private:
  eoEvalFunc<EOT> & eval;
};

#ifdef WITH_MPI
/**
 * @brief Evaluator of a population of EOT which uses parallelization to evaluate individuals.
 *
 * This class implements an instance of eoPopEvalFunc that applies a private eoEvalFunc to
 * all offspring, but in a parallel way. The original process becomes the central host from a network ("master"), and
 * other machines disponible in the MPI network ("slaves") are used as evaluators. Population to evaluate is splitted in
 * little packets of individuals, which are sent to the evaluators, that process the effective call to eval. Once all
 * the individuals have been evaluated, they are returned to the master. The whole process is entirely invisible to the
 * eyes of the user, who just has to launch a certain number of processes in MPI so as to have a result.
 *
 * The eoEvalFunc is no more directly given, but it is stored in the eo::mpi::ParallelApplyStore, which can be
 * instanciated if no one is given at construction.
 *
 * The use of this class requires the user to have called the eo::mpi::Node::init function, at the beginning of its
 * program.
 *
 * @ingroup Evaluation Parallel
 *
 * @author Benjamin Bouvier <benjamin.bouvier@gmail.com>
 */
template<class EOT>
class eoParallelPopLoopEval : public eoPopEvalFunc<EOT>
{
    public:
        /**
         * @brief Constructor which creates the job store for the user.
         *
         * This constructor is the simplest to use, as it creates the store used by the parallel job, for the user.
         * The user just precises the scheduling algorithm, the rank of the master and then gives its eval function and
         * the size of a packet (how many individuals should be in a single message to evaluator).
         *
         * @param _assignAlgo The scheduling algorithm used to give orders to evaluators.
         * @param _masterRank The MPI rank of the master.
         * @param _eval The evaluation functor used to evaluate each individual in the population.
         * @param _packetSize The number of individuals to send in one message to evaluator, and which are evaluated at
         * a time.
         */
        eoParallelPopLoopEval(
                // Job parameters
                eo::mpi::AssignmentAlgorithm& _assignAlgo,
                int _masterRank,
                // Default parameters for store
                eoEvalFunc<EOT> & _eval,
                int _packetSize = 1
                ) :
            assignAlgo( _assignAlgo ),
            masterRank( _masterRank ),
            needToDeleteStore( true ) // we used new, we'll have to use delete (RAII)
        {
            store = new eo::mpi::ParallelApplyStore<EOT>( _eval, _masterRank, _packetSize );
        }

        /**
         * @brief Constructor which allows the user to customize its job store.
         *
         * This constructor allows the user to customize the store, for instance by adding wrappers and other
         * functionnalities, before using it for the parallelized evaluation.
         *
         * @param _assignAlgo The scheduling algorithm used to give orders to evaluators.
         * @param _masterRank The MPI rank of the master.
         * @param _store Pointer to a parallel eval store given by the user.
         */
        eoParallelPopLoopEval(
                // Job parameters
                eo::mpi::AssignmentAlgorithm& _assignAlgo,
                int _masterRank,
                eo::mpi::ParallelApplyStore<EOT>* _store
                ) :
            assignAlgo( _assignAlgo ),
            masterRank( _masterRank ),
            store( _store ),
            needToDeleteStore( false ) // we haven't used new for creating store, we don't care if we have to delete it (RAII).
        {
            // empty
        }

        /**
         * @brief Default destructor. Sends a message to all evaluators indicating that the global loop (eoEasyEA, for
         * instance) is over.
         */
        ~eoParallelPopLoopEval()
        {
            // Only the master has to send the termination message
            if( eo::mpi::Node::comm().rank() == masterRank )
            {
                eo::mpi::EmptyJob job( assignAlgo, masterRank );
                job.run();
            }

            // RAII
            if( needToDeleteStore )
            {
                delete store;
            }
        }

        /**
         * @brief Parallel implementation of the operator().
         *
         * @param _parents Population of parents (ignored).
         * @param _offspring Population of children, which will be evaluated.
         */
        void operator()( eoPop<EOT> & _parents, eoPop<EOT> & _offspring )
        {
            (void)_parents;
            // Reinits the store and the scheduling algorithm
            store->data( _offspring );
            // For static scheduling, it's mandatory to reinit attributions
            int nbWorkers = assignAlgo.availableWorkers();
            assignAlgo.reinit( nbWorkers );
            if( ! eo::parallel.isDynamic() ) {
                store->data()->packetSize = ceil( static_cast<double>( _offspring.size() ) / nbWorkers );
            }
            // Effectively launches the job.
            eo::mpi::ParallelApply<EOT> job( assignAlgo, masterRank, *store );
            job.run();
        }

    private:
        // Scheduling algorithm
        eo::mpi::AssignmentAlgorithm & assignAlgo;
        // Master MPI rank
        int masterRank;

        // Store
        eo::mpi::ParallelApplyStore<EOT>* store;
        // Do we have to delete the store by ourselves ?
        bool needToDeleteStore;
};

/**
 * @example t-mpi-eval.cpp
 */
#endif

/////////////////////////////////////////////////////////////
//           eoTimeVaryingLoopEval
/////////////////////////////////////////////////////////////

/** eoPopLoopEval: an instance of eoPopEvalFunc that simply applies
 *     a private eoEvalFunc to all offspring AND ALL PARENTS
 *     as the fitness is supposed here to vary
 *
 *    @ingroup Evaluation
 */
template<class EOT>
class eoTimeVaryingLoopEval : public eoPopEvalFunc<EOT> {
public:
  /** Ctor: set value of embedded eoEvalFunc */
  eoTimeVaryingLoopEval(eoEvalFunc<EOT> & _eval) : eval(_eval) {}

  /** Do the job: simple loop over the offspring */
  void operator()(eoPop<EOT> & _parents, eoPop<EOT> & _offspring)
  {
    apply<EOT>(eval, _parents);
    apply<EOT>(eval, _offspring);
  }

private:
  eoEvalFunc<EOT> & eval;
};

#endif
