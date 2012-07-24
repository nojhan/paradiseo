# include "eoMpi.h"

namespace eo
{
    namespace mpi
    {
        /**********************************************
         * *********** GLOBALS ************************
         * *******************************************/
        eoTimerStat timerStat;

        namespace Channel
        {
            const int Commands = 0;
            const int Messages = 1;
        }

        namespace Message
        {
            const int Continue = 0;
            const int Finish = 1;
            const int Kill = 2;
        }

        const int DEFAULT_MASTER = 0;
    }
}
