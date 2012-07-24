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
# ifndef __MPI_NODE_H__
# define __MPI_NODE_H__

# include "implMpi.h"
namespace bmpi = mpi;

namespace eo
{
    namespace mpi
    {
        /**
         * @brief Global object used to reach mpi::communicator everywhere.
         *
         * mpi::communicator is the main object used to send and receive messages between the different hosts of
         * a MPI algorithm.
         *
         * @ingroup MPI
         */
        class Node
        {
            public:

                /**
                 * @brief Initializes the MPI environment with argc and argv.
                 *
                 * Should be called at the beginning of every parallel program.
                 *
                 * @param argc Main's argc
                 * @param argv Main's argv
                 */
                static void init( int argc, char** argv );

                /**
                 * @brief Returns the global mpi::communicator
                 */
                static bmpi::communicator& comm();

            protected:
                static bmpi::communicator _comm;
        };
    }
}
# endif // __MPI_NODE_H__

