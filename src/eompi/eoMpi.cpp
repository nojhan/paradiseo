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
