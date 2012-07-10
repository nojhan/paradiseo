# ifndef __MPI_ASSIGNMENT_ALGORITHM_H__
# define __MPI_ASSIGNMENT_ALGORITHM_H__

# include <vector>
# include "eoMpiNode.h"

namespace eo
{
    namespace mpi
    {
        const int REST_OF_THE_WORLD = -1;

        struct AssignmentAlgorithm
        {
            virtual int get( ) = 0;
            virtual int availableWorkers( ) = 0;
            virtual void confirm( int wrkRank ) = 0;
            virtual std::vector<int> idles( ) = 0;
            virtual void reinit( int runs ) = 0;
        };

        struct DynamicAssignmentAlgorithm : public AssignmentAlgorithm
        {
            public:

                DynamicAssignmentAlgorithm( )
                {
                    for(int i = 1; i < Node::comm().size(); ++i)
                    {
                        availableWrk.push_back( i );
                    }
                }

                DynamicAssignmentAlgorithm( int unique )
                {
                    availableWrk.push_back( unique );
                }

                DynamicAssignmentAlgorithm( const std::vector<int> & workers )
                {
                    availableWrk = workers;
                }

                DynamicAssignmentAlgorithm( int first, int last )
                {
                    if( last == REST_OF_THE_WORLD )
                    {
                        last = Node::comm().size() - 1;
                    }

                    for( int i = first; i <= last; ++i)
                    {
                        availableWrk.push_back( i );
                    }
                }

                virtual int get( )
                {
                    int assignee = -1;
                    if (! availableWrk.empty() )
                    {
                        assignee = availableWrk.back();
                        availableWrk.pop_back();
                    }
                    return assignee;
                }

                int availableWorkers()
                {
                    return availableWrk.size();
                }

                void confirm( int rank )
                {
                    availableWrk.push_back( rank );
                }

                std::vector<int> idles( )
                {
                    return availableWrk;
                }

                void reinit( int _ )
                {
                    ++_;
                    // nothing to do
                }

            protected:
                std::vector< int > availableWrk;
        };

        struct StaticAssignmentAlgorithm : public AssignmentAlgorithm
        {
            public:
                StaticAssignmentAlgorithm( std::vector<int>& workers, int runs )
                {
                    init( workers, runs );
                }

                StaticAssignmentAlgorithm( int first, int last, int runs )
                {
                    std::vector<int> workers;

                    if( last == REST_OF_THE_WORLD )
                    {
                        last = Node::comm().size() - 1;
                    }

                    for(int i = first; i <= last; ++i)
                    {
                        workers.push_back( i );
                    }
                    init( workers, runs );
                }

                StaticAssignmentAlgorithm( int runs )
                {
                    std::vector<int> workers;
                    for(int i = 1; i < Node::comm().size(); ++i)
                    {
                        workers.push_back( i );
                    }
                    init( workers, runs );
                }

                StaticAssignmentAlgorithm( int unique, int runs )
                {
                    std::vector<int> workers;
                    workers.push_back( unique );
                    init( workers, runs );
                }

            private:
                void init( std::vector<int> & workers, int runs )
                {
                    unsigned int nbWorkers = workers.size();
                    freeWorkers = nbWorkers;
                    attributions.reserve( nbWorkers );
                    busy.resize( nbWorkers, false );

                    // Let be the euclidean division of runs by nbWorkers :
                    // runs == q * nbWorkers + r, 0 <= r < nbWorkers
                    // This one liner affects q requests to each worker
                    for (unsigned int i = 0; i < nbWorkers; attributions[i++] = runs / nbWorkers) ;
                    // The first line computes r and the one liner affects the remaining
                    // r requests to workers, in ascending order
                    unsigned int diff = runs - (runs / nbWorkers) * nbWorkers;
                    for (unsigned int i = 0; i < diff; ++attributions[i++]);

                    realRank = workers;
                }

            public:
                int get( )
                {
                    int assignee = -1;
                    for( unsigned i = 0; i < busy.size(); ++i )
                    {
                        if( !busy[i] && attributions[i] > 0 )
                        {
                            busy[i] = true;
                            --freeWorkers;
                            assignee = realRank[ i ];
                            break;
                        }
                    }
                    return assignee;
                }

                int availableWorkers( )
                {
                    return freeWorkers;
                }

                std::vector<int> idles()
                {
                    std::vector<int> ret;
                    for(unsigned int i = 0; i < busy.size(); ++i)
                    {
                        if( !busy[i] )
                        {
                            ret.push_back( realRank[i] );
                        }
                    }
                    return ret;
                }

                void confirm( int rank )
                {
                    int i = -1; // i is the real index in table
                    for( unsigned int j = 0; j < realRank.size(); ++j )
                    {
                        if( realRank[j] == rank )
                        {
                            i = j;
                            break;
                        }
                    }

                    --attributions[ i ];
                    busy[ i ] = false;
                    ++freeWorkers;
                }

                void reinit( int runs )
                {
                    init( realRank, runs );
                }

            private:
                std::vector<int> attributions;
                std::vector<int> realRank;
                std::vector<bool> busy;
                unsigned int freeWorkers;
        };
    }
}
# endif // __MPI_ASSIGNMENT_ALGORITHM_H__
