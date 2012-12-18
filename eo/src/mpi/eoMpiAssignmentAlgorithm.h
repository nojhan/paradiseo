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
# ifndef __MPI_ASSIGNMENT_ALGORITHM_H__
# define __MPI_ASSIGNMENT_ALGORITHM_H__

# include <vector> // std::vector

namespace eo
{
    namespace mpi
    {
        /**
         * @brief Constant indicating to use all the resting available workers, in assignment algorithms constructor
         * using an interval.
         *
         * @ingroup MPI
         */
        extern const int REST_OF_THE_WORLD;

        /**
         * @brief Contains informations on the available workers and allows to find assignees for jobs.
         *
         * Available workers are workers who aren't processing anything. When they've received an order, workers switch
         * from the state "available" to the state "busy", and the master has to wait for their response for considering
         * them available again.
         *
         * @ingroup MPI
         */
        struct AssignmentAlgorithm
        {
            /**
             * @brief Gets the rank of an available worker, so as to send it a task.
             *
             * @return The MPI rank of an available worker, or -1 if there is no available worker.
             */
            virtual int get( ) = 0;

            /**
             * @brief Gets the number of total available workers.
             *
             * Before the first call, it is equal to the total number of present workers, as specified in the
             * specific assignment algorithm constructor. It allows the Job class to know when all the responses have
             * been received, by comparing this number to the total number of workers.
             *
             * @return Integer indicating how many workers are available.
             */
            virtual int availableWorkers( ) = 0;

            /**
             * @brief Reinject the worker of indicated rank in the available state.
             *
             * @param wrkRank The MPI rank of the worker who has finished its job.
             */
            virtual void confirm( int wrkRank ) = 0;

            /**
             * @brief Indicates who are the workers which do nothing.
             *
             * At the end of the algorithm, the master has to warn all the workers that it's done. All the workers mean,
             * the workers which are currently processing data, and the other ones who could be waiting : the idles.
             * This function indicates to the master which worker aren't doing anything.
             *
             * @return A std::vector containing all the MPI ranks of the idles workers.
             */
            virtual std::vector<int> idles( ) = 0;

            /**
             * @brief Reinitializes the assignment algorithm with the right number of runs.
             *
             * In fact, this is only useful for static assignment algorithm, which has to be reinitialized every time
             * it's used, in the case of a Multi Job. It's the user's responsability to call this function.
             *
             * @todo Not really clean. Find a better way to do it.
             */
            virtual void reinit( int runs ) = 0;
        };

        /**
         * @brief Assignment (scheduling) algorithm which handles workers in a queue.
         *
         * With this assignment algorithm, workers are put in a queue and may be called an unlimited number of times.
         * Whenever a worker returns, it is added to the queue, and it becomes available for the next call to get().
         * The available workers are all located in the queue at any time, so the number of available workers is
         * directly equal to the size of the queue.
         *
         * This kind of assignment is adapted for tasks whose execution time is stochastic or unknown, but without any
         * warranty to be faster than other assignments.
         *
         * @ingroup MPI
         */
        struct DynamicAssignmentAlgorithm : public AssignmentAlgorithm
        {
            public:

                /**
                 * @brief Uses all the hosts whose rank is higher to 1, inclusive, as workers.
                 */
                DynamicAssignmentAlgorithm( );

                /**
                 * @brief Uses the unique host with given rank as a worker.
                 *
                 * @param unique MPI rank of the unique worker.
                 */
                DynamicAssignmentAlgorithm( int unique );

                /**
                 * @brief Uses the workers whose ranks are present in the argument as workers.
                 *
                 * @param workers std::vector containing MPI ranks of workers.
                 */
                DynamicAssignmentAlgorithm( const std::vector<int> & workers );

                /**
                 * @brief Uses a range of ranks as workers.
                 *
                 * @param first The first worker to be included (inclusive)
                 * @param last The last worker to be included (inclusive). If last == eo::mpi::REST_OF_THE_WORLD, all
                 * hosts whose rank is higher than first are taken.
                 */
                DynamicAssignmentAlgorithm( int first, int last );

                virtual int get( );

                int availableWorkers();

                void confirm( int rank );

                std::vector<int> idles( );

                void reinit( int _ );

            protected:
                std::vector< int > availableWrk;
        };

        /**
         * @brief Assignment algorithm which gives to each worker a precise number of tasks to do, in a round robin
         * fashion.
         *
         * This scheduling algorithm attributes, at initialization or when calling reinit(), a fixed amount of runs to
         * distribute to the workers. The amount of runs is then equally distributed between all workers ; if total
         * number of runs is not a direct multiple of workers number, then remainding unaffected runs are distributed to
         * workers from the first to the last, in a round-robin fashion.
         *
         * This scheduling should be used when the amount of runs can be computed or is fixed and when we guess that the
         * duration of processing task will be the same for each run. There is no warranty that this algorithm is more
         * or less efficient that another one. When having a doubt, use DynamicAssignmentAlgorithm.
         *
         * @ingroup MPI
         */
        struct StaticAssignmentAlgorithm : public AssignmentAlgorithm
        {
            public:
                /**
                 * @brief Uses a given precise set of workers.
                 *
                 * @param workers std::vector of MPI ranks of workers which will be used.
                 * @param runs Fixed amount of runs, strictly positive.
                 */
                StaticAssignmentAlgorithm( const std::vector<int>& workers, int runs );

                /**
                 * @brief Uses a range of workers.
                 *
                 * @param first The first MPI rank of worker to use
                 * @param last The last MPI rank of worker to use. If it's equal to REST_OF_THE_WORLD, then all the
                 * workers from the first one are taken as workers.
                 * @param runs Fixed amount of runs, strictly positive.
                 */
                StaticAssignmentAlgorithm( int first, int last, int runs );

                /**
                 * @brief Uses all the hosts whose rank is higher than 1 (inclusive) as workers.
                 *
                 * @param runs Fixed amount of runs, strictly positive. If it's not set, you'll have to call reinit()
                 * later.
                 */
                StaticAssignmentAlgorithm( int runs = 0 );

                /**
                 * @brief Uses an unique host as worker.
                 *
                 * @param unique The MPI rank of the host which will be the worker.
                 * @param runs Fixed amount of runs, strictly positive.
                 */
                StaticAssignmentAlgorithm( int unique, int runs );

            private:
                /**
                 * @brief Initializes the static scheduling.
                 *
                 * Gives to each worker an equal attribution number, equal to runs / workers.size(), eventually plus one
                 * if number of workers is not a divisor of runs.
                 *
                 * @param workers Vector of hosts' ranks
                 * @param runs Fixed amount of runs, strictly positive.
                 */
                void init( const std::vector<int> & workers, int runs );

            public:
                int get( );

                int availableWorkers( );

                std::vector<int> idles();

                void confirm( int rank );

                void reinit( int runs );

            private:
                std::vector<int> attributions;
                std::vector<int> realRank;
                std::vector<bool> busy;
                unsigned int freeWorkers;
        };
    }
}
# endif // __MPI_ASSIGNMENT_ALGORITHM_H__
