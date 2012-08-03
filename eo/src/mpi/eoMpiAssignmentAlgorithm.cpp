# include "eoMpiAssignmentAlgorithm.h"
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
# include "eoMpiNode.h"

namespace eo
{
    namespace mpi
    {
        const int REST_OF_THE_WORLD = -1;

        /********************************************************
         * DYNAMIC ASSIGNMENT ALGORITHM *************************
         *******************************************************/

        DynamicAssignmentAlgorithm::DynamicAssignmentAlgorithm( )
        {
            for(int i = 1; i < Node::comm().size(); ++i)
            {
                availableWrk.push_back( i );
            }
        }

        DynamicAssignmentAlgorithm::DynamicAssignmentAlgorithm( int unique )
        {
            availableWrk.push_back( unique );
        }

        DynamicAssignmentAlgorithm::DynamicAssignmentAlgorithm( const std::vector<int> & workers )
        {
            availableWrk = workers;
        }

        DynamicAssignmentAlgorithm::DynamicAssignmentAlgorithm( int first, int last )
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

        int DynamicAssignmentAlgorithm::get( )
        {
            int assignee = -1;
            if (! availableWrk.empty() )
            {
                assignee = availableWrk.back();
                availableWrk.pop_back();
            }
            return assignee;
        }

        int DynamicAssignmentAlgorithm::availableWorkers()
        {
            return availableWrk.size();
        }

        void DynamicAssignmentAlgorithm::confirm( int rank )
        {
            availableWrk.push_back( rank );
        }

        std::vector<int> DynamicAssignmentAlgorithm::idles( )
        {
            return availableWrk;
        }

        void DynamicAssignmentAlgorithm::reinit( int _ )
        {
            ++_;
            // nothing to do
        }

        /********************************************************
         * STATIC ASSIGNMENT ALGORITHM **************************
         *******************************************************/

        StaticAssignmentAlgorithm::StaticAssignmentAlgorithm( const std::vector<int>& workers, int runs )
        {
            init( workers, runs );
        }

        StaticAssignmentAlgorithm::StaticAssignmentAlgorithm( int first, int last, int runs )
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

        StaticAssignmentAlgorithm::StaticAssignmentAlgorithm( int runs )
        {
            std::vector<int> workers;
            for(int i = 1; i < Node::comm().size(); ++i)
            {
                workers.push_back( i );
            }

            init( workers, runs );
        }

        StaticAssignmentAlgorithm::StaticAssignmentAlgorithm( int unique, int runs )
        {
            std::vector<int> workers;
            workers.push_back( unique );
            init( workers, runs );
        }

        void StaticAssignmentAlgorithm::init( const std::vector<int> & workers, int runs )
        {
            unsigned int nbWorkers = workers.size();
            freeWorkers = nbWorkers;

            busy.clear();
            busy.resize( nbWorkers, false );
            realRank = workers;

            if( runs <= 0 )
            {
                return;
            }

            attributions.clear();
            attributions.reserve( nbWorkers );

            // Let be the euclidean division of runs by nbWorkers :
            // runs == q * nbWorkers + r, 0 <= r < nbWorkers
            // This one liner affects q requests to each worker
            for (unsigned int i = 0; i < nbWorkers; attributions[i++] = runs / nbWorkers) ;
            // The first line computes r and the one liner affects the remaining
            // r requests to workers, in ascending order
            unsigned int diff = runs - (runs / nbWorkers) * nbWorkers;
            for (unsigned int i = 0; i < diff; ++attributions[i++]);
        }

        int StaticAssignmentAlgorithm::get( )
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

        int StaticAssignmentAlgorithm::availableWorkers( )
        {
            return freeWorkers;
        }

        std::vector<int> StaticAssignmentAlgorithm::idles()
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

        void StaticAssignmentAlgorithm::confirm( int rank )
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

        void StaticAssignmentAlgorithm::reinit( int runs )
        {
            init( realRank, runs );
        }
    }
}
