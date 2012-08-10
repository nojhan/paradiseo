/*
(c) Thales group, 2012

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation;
    version 2 of the License.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
Contact: http://eodev.sourceforge.net

Authors:
    Benjamin Bouvier <benjamin.bouvier@gmail.com>
*/
# ifndef __EO_PARALLEL_APPLY_H__
# define __EO_PARALLEL_APPLY_H__

# include "eoMpi.h"

# include <eoFunctor.h> // eoUF
# include <vector> // std::vector population

/**
 * @file eoParallelApply.h
 *
 * @brief Applies a functor with single parameter to elements of a table, in a parallel fashion.
 *
 * This file contains all the required classes to do a parallel apply of a table, in a parallel fashion. This can be
 * very useful when applying the function can be made without any dependances within the data. In EO, it occurs in
 * particular during the evaluation: the number of individuals to evaluate can be really high, and the evaluation of one
 * individual is independant from the evaluation of other individuals.
 *
 * Elements in the table are directly replaced, as the table is given by reference. No new table is made during the
 * process.
 *
 * User can tune this job, indicating how many elements of the table should be sent and evaluated by a worker, at a
 * time: this is called the "packet size", as individuals are groupped into a packet of individuals which are sent to
 * the worker before evaluation. The problem of choosing the optimal packet size is beyond the purposes of this documentation 
 * and deserves a theoritical study.
 *
 * This job is the parallel equivalent to the function apply<EOT>, defined in apply.h. It just applies the function to
 * every element of a table. In Python or Javascript, it's the equivalent of the function Map.
 */

namespace eo
{
    namespace mpi
    {
        /**
         * @brief Structure used to save assignment to a worker, i.e which slice of the table it has to process.
         *
         * This slice is defined by the index of the first evaluated argument and the number of processed elements.
         */
        struct ParallelApplyAssignment
        {
            int index;
            int size;
        };

        /**
         * @brief Data useful for a parallel apply (map).
         *
         * A parallel apply needs at least the functor to apply to every element of the table, and the table itself,
         * whereas it can be set later with the function init(). Master rank is also needed, to send it informations and
         * receive informations from it, inside the functors (the job knows these values, but the functors don't). The
         * size of a packet can be tuned here.
         *
         * Internal attributes contain:
         * - (useful for master) the index of the next element to be evaluated.
         * - (useful for master) a map containing links between MPI ranks and slices of the table which the worker with
         *   this rank has evaluated. Without this map, when receiving results from a worker, the master couldn't be
         *   able to replace the right elements in the table.
         *
         * @ingroup MPI
         */
        template<class EOT>
        struct ParallelApplyData
        {
            /**
             * @brief Ctor for Parallel apply (map) data.
             *
             * @param _proc The functor to apply on each element in the table
             * @param _masterRank The MPI rank of the master
             * @param _packetSize The number of elements on which the function will be applied by the worker, at a time.
             * @param table The table to apply. If this value is NULL, user will have to call init() before launching the
             * job.
             */
            ParallelApplyData(
                    eoUF<EOT&, void> & _proc,
                    int _masterRank,
                    int _packetSize,
                    std::vector<EOT> * table = 0
                   ) :
                _table( table ), func( _proc ), index( 0 ), packetSize( _packetSize ), masterRank( _masterRank ), comm( Node::comm() )
            {
                if ( _packetSize <= 0 )
                {
                    throw std::runtime_error("Packet size should not be negative.");
                }

                if( table )
                {
                    size = table->size();
                }
            }

            /**
             * @brief Reinitializes the data for a new table to evaluate.
             */
            void init( std::vector<EOT>& table )
            {
                index = 0;
                size = table.size();
                _table = &table;
                assignedTasks.clear();
            }

            std::vector<EOT>& table()
            {
                return *_table;
            }

            // All elements are public since functors will often use them.
            std::vector<EOT> * _table;
            eoUF<EOT&, void> & func;
            int index;
            int size;
            std::map< int /* worker rank */, ParallelApplyAssignment /* last assignment */> assignedTasks;
            int packetSize;
            std::vector<EOT> tempArray;

            int masterRank;
            bmpi::communicator& comm;
        };

        /**
         * @brief Send task functor implementation for the parallel apply (map) job.
         *
         * Master side: Sends a slice of the table to evaluate to the worker.
         *
         * Implementation details:
         * Finds the next slice of data to send to the worker, sends first the size and then the data, and memorizes
         * that this slice has been distributed to the worker, then updates the next position of element to evaluate.
         */
        template< class EOT >
        class SendTaskParallelApply : public SendTaskFunction< ParallelApplyData<EOT> >
        {
            public:
            using SendTaskFunction< ParallelApplyData<EOT> >::_data;

            SendTaskParallelApply( SendTaskParallelApply<EOT> * w = 0 ) : SendTaskFunction< ParallelApplyData<EOT> >( w )
            {
                // empty
            }

            void operator()(int wrkRank)
            {
                int futureIndex;

                if( _data->index + _data->packetSize < _data->size )
                {
                    futureIndex = _data->index + _data->packetSize;
                } else {
                    futureIndex = _data->size;
                }

                int sentSize = futureIndex - _data->index ;

                _data->comm.send( wrkRank, eo::mpi::Channel::Messages, sentSize );

                eo::log << eo::debug << "Evaluating individual " << _data->index << std::endl;

                _data->assignedTasks[ wrkRank ].index = _data->index;
                _data->assignedTasks[ wrkRank ].size = sentSize;

                _data->comm.send( wrkRank, eo::mpi::Channel::Messages, & ( (_data->table())[ _data->index ] ) , sentSize );
                _data->index = futureIndex;
            }
        };

        /**
         * @brief Handle response functor implementation for the parallel apply (map) job.
         *
         * Master side: Replaces the slice of data attributed to the worker in the table.
         */
        template< class EOT >
        class HandleResponseParallelApply : public HandleResponseFunction< ParallelApplyData<EOT> >
        {
            public:
            using HandleResponseFunction< ParallelApplyData<EOT> >::_data;

            HandleResponseParallelApply( HandleResponseParallelApply<EOT> * w = 0 ) : HandleResponseFunction< ParallelApplyData<EOT> >( w )
            {
                // empty
            }

            void operator()(int wrkRank)
            {
                _data->comm.recv( wrkRank, eo::mpi::Channel::Messages, & (_data->table()[ _data->assignedTasks[wrkRank].index ] ), _data->assignedTasks[wrkRank].size );
            }
        };

        /**
         * @brief Process task functor implementation for the parallel apply (map) job.
         *
         * Worker side: apply the function to the given slice of data.
         *
         * Implementation details: retrieves the number of elements to evaluate, retrieves them, applies the function
         * and then returns the results.
         */
        template< class EOT >
        class ProcessTaskParallelApply : public ProcessTaskFunction< ParallelApplyData<EOT> >
        {
            public:
            using ProcessTaskFunction< ParallelApplyData<EOT> >::_data;

            ProcessTaskParallelApply( ProcessTaskParallelApply<EOT> * w = 0 ) : ProcessTaskFunction< ParallelApplyData<EOT> >( w )
            {
                // empty
            }

            void operator()()
            {
                int recvSize;

                _data->comm.recv( _data->masterRank, eo::mpi::Channel::Messages, recvSize );
                _data->tempArray.resize( recvSize );
                _data->comm.recv( _data->masterRank, eo::mpi::Channel::Messages, & _data->tempArray[0] , recvSize );
                timerStat.start("worker_processes");
                for( int i = 0; i < recvSize ; ++i )
                {
                    _data->func( _data->tempArray[ i ] );
                }
                timerStat.stop("worker_processes");
                _data->comm.send( _data->masterRank, eo::mpi::Channel::Messages, & _data->tempArray[0], recvSize );
            }
        };

        /**
         * @brief Is finished functor implementation for the parallel apply (map) job.
         *
         * Master side: returns true if and only if the whole table has been evaluated. The job is also terminated only
         * when the whole table has been evaluated.
         */
        template< class EOT >
        class IsFinishedParallelApply : public IsFinishedFunction< ParallelApplyData<EOT> >
        {
            public:
            using IsFinishedFunction< ParallelApplyData<EOT> >::_data;

            IsFinishedParallelApply( IsFinishedParallelApply<EOT> * w = 0 ) : IsFinishedFunction< ParallelApplyData<EOT> >( w )
            {
                // empty
            }

            bool operator()()
            {
                return _data->index == _data->size;
            }
        };

        /**
         * @brief Store containing all the datas and the functors for the parallel apply (map) job.
         *
         * User can tune functors when constructing the object. For each functor which is not given, a default one is
         * generated.
         *
         * @ingroup MPI
         */
        template< class EOT >
        struct ParallelApplyStore : public JobStore< ParallelApplyData<EOT> >
        {
            using JobStore< ParallelApplyData<EOT> >::_stf;
            using JobStore< ParallelApplyData<EOT> >::_hrf;
            using JobStore< ParallelApplyData<EOT> >::_ptf;
            using JobStore< ParallelApplyData<EOT> >::_iff;

            /**
             * @brief Main constructor for the parallel apply (map) job.
             *
             * @param _proc The procedure to apply to each element of the table.
             * @param _masterRank The rank of the master process.
             * @param _packetSize The number of elements of the table to be evaluated at a time, by the worker.
             * @param stpa Pointer to Send Task parallel apply functor descendant. If null, a default one is used.
             * @param hrpa Pointer to Handle Response parallel apply functor descendant. If null, a default one is used.
             * @param ptpa Pointer to Process Task parallel apply functor descendant. If null, a default one is used.
             * @param ifpa Pointer to Is Finished parallel apply functor descendant. If null, a default one is used.
             */
            ParallelApplyStore(
                    eoUF<EOT&, void> & _proc,
                    int _masterRank,
                    int _packetSize = 1,
                    // JobStore functors
                    SendTaskParallelApply<EOT> * stpa = 0,
                    HandleResponseParallelApply<EOT>* hrpa = 0,
                    ProcessTaskParallelApply<EOT>* ptpa = 0,
                    IsFinishedParallelApply<EOT>* ifpa = 0
                   ) :
                _data( _proc, _masterRank, _packetSize )
            {
                if( stpa == 0 ) {
                    stpa = new SendTaskParallelApply<EOT>;
                    stpa->needDelete( true );
                }

                if( hrpa == 0 ) {
                    hrpa = new HandleResponseParallelApply<EOT>;
                    hrpa->needDelete( true );
                }

                if( ptpa == 0 ) {
                    ptpa = new ProcessTaskParallelApply<EOT>;
                    ptpa->needDelete( true );
                }

                if( ifpa == 0 ) {
                    ifpa = new IsFinishedParallelApply<EOT>;
                    ifpa->needDelete( true );
                }

                _stf = stpa;
                _hrf = hrpa;
                _ptf = ptpa;
                _iff = ifpa;
            }

            ParallelApplyData<EOT>* data() { return &_data; }

            /**
             * @brief Reinits the store with a new table to evaluate.
             *
             * @param _pop The table of elements to be evaluated.
             */
            void data( std::vector<EOT>& _pop )
            {
                _data.init( _pop );
            }

            virtual ~ParallelApplyStore() // for inheritance purposes only
            {
            }

            protected:
            ParallelApplyData<EOT> _data;
        };

        /**
         * @brief Parallel apply job. Present for convenience only.
         *
         * A typedef wouldn't have been working, as typedef on templates don't work in C++. Traits would be a
         * disgraceful overload for the user.
         *
         * @ingroup MPI
         * @see eoParallelApply.h
         */
        template< typename EOT >
        class ParallelApply : public MultiJob< ParallelApplyData<EOT> >
        {
            public:

            ParallelApply(
                    AssignmentAlgorithm & algo,
                    int _masterRank,
                    ParallelApplyStore<EOT> & store
                    ) :
                MultiJob< ParallelApplyData<EOT> >( algo, _masterRank, store )
            {
                // empty
            }
        };

        /**
         * @example t-mpi-parallelApply.cpp
         * @example t-mpi-multipleRoles.cpp
         */
    }
}
# endif // __EO_PARALLEL_APPLY_H__


