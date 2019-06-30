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
#ifndef __EO_TERMINATE_H__
#define __EO_TERMINATE_H__

#include "eoMpi.h"

namespace eo
{
    namespace mpi
    {
        /**
         * @ingroup MPI
         * @{
         */

        /**
         * @brief Send task functor which does nothing.
         */
        struct DummySendTaskFunction : public SendTaskFunction<void>
        {
            void operator()( int _ )
            {
                ++_;
            }
        };

        /**
         * @brief Handle response functor which does nothing.
         */
        struct DummyHandleResponseFunction : public HandleResponseFunction<void>
        {
            void operator()( int _ )
            {
                ++_;
            }
        };

        /**
         * @brief Process task functor which does nothing.
         */
        struct DummyProcessTaskFunction : public ProcessTaskFunction<void>
        {
            void operator()()
            {
                // nothing!
            }
        };

        /**
         * @brief Is finished functor which returns true everytime.
         */
        struct DummyIsFinishedFunction : public IsFinishedFunction<void>
        {
            bool operator()()
            {
                return true;
            }
        };

        /**
         * @brief Job store containing all dummy functors and containing no data.
         */
        struct DummyJobStore : public JobStore<void>
        {
            using JobStore<void>::_stf;
            using JobStore<void>::_hrf;
            using JobStore<void>::_ptf;
            using JobStore<void>::_iff;

            DummyJobStore()
            {
                _stf = new DummySendTaskFunction;
                _stf->needDelete( true );
                _hrf = new DummyHandleResponseFunction;
                _hrf->needDelete( true );
                _ptf = new DummyProcessTaskFunction;
                _ptf->needDelete( true );
                _iff = new DummyIsFinishedFunction;
                _iff->needDelete( true );
            }

            void* data() { return 0; }
        };

        /**
         * @brief Job to run after a Multi Job, so as to indicate that every workers should terminate.
         */
        struct EmptyJob : public OneShotJob<void>
        {
            /**
             * @brief Main EmptyJob ctor
             *
             * @param algo Assignment (scheduling) algorithm used.
             * @param masterRank The rank of the master process.
             */
            EmptyJob( AssignmentAlgorithm& algo, int masterRank ) :
                OneShotJob<void>( algo, masterRank, *(new DummyJobStore) )
                // the job store is deleted on destructor
            {
                // empty
            }

            ~EmptyJob()
            {
                std::vector< int > idles = assignmentAlgo.idles();
                for(unsigned i = 0, size = idles.size(); i < size; ++i)
                {
                    comm.send( idles[i], Channel::Commands, Message::Kill );
                }
                delete & this->store;
            }
        };

        /**
         * @}
         */
    }
}

# endif // __EO_TERMINATE_H__
