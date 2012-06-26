# ifndef __EO_TERMINATE_H__
# define __EO_TERMINATE_H__

# include "eoMpi.h"

namespace eo
{
    namespace mpi
    {
        class TerminateJob : public Job
        {
            public:
                TerminateJob( AssignmentAlgorithm& algo, int _ )
                    : Job( algo, _ )
                {
                    // empty
                }

                void sendTask( int wrkRank )
                {
                    // empty
                }

                void handleResponse( int wrkRank )
                {
                    // empty
                }

                void processTask( )
                {
                    // empty
                }

                bool isFinished()
                {
                    return true;
                }
        };
    }
}

# endif // __EO_TERMINATE_H__
