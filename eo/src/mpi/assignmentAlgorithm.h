# ifndef __ASSIGNMENT_ALGORITHM_H__
# define __ASSIGNMENT_ALGORITHM_H__

# include <vector>

struct AssignmentAlgorithm
{
    virtual int get( ) = 0;
    virtual int availableWorkers( ) = 0;
    virtual void confirm( int wrkRank ) = 0;
    virtual std::vector<int> idles( ) = 0;
};

struct DynamicAssignmentAlgorithm : public AssignmentAlgorithm
{
    public:
        DynamicAssignmentAlgorithm( int first, int last )
        {
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

    protected:
        std::vector< int > availableWrk;
};

struct StaticAssignmentAlgorithm : public AssignmentAlgorithm
{
    public:
        StaticAssignmentAlgorithm( int first, int last, int runs )
        {
            unsigned int nbWorkers = last - first + 1;
            freeWorkers = nbWorkers;
            offset = first;
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
        }

        int get( )
        {
            int assignee = -1;
            for( unsigned i = 0; i < busy.size(); ++i )
            {
                if( !busy[i] && attributions[i] > 0 )
                {
                    busy[i] = true;
                    --freeWorkers;
                    assignee = realRank( i );
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
                    eo::log << "Idle : " << realRank(i) <<
                        " / attributions : " << attributions[i] << std::endl;
                    ret.push_back( realRank(i) );
                }
            }
            afterIdle = true;
            return ret;
        }

        void confirm( int rank )
        {
            int i = attributionsIndex( rank );
            --attributions[ i ];
            busy[ i ] = false;
            ++freeWorkers;
        }

    private:
        int attributionsIndex( int rank )
        {
            return rank - offset;
        }

        int realRank( int index )
        {
            return index + offset;
        }

        std::vector<int> attributions;
        std::vector<bool> busy;

        bool afterIdle;
        int runs;
        int offset;
        unsigned int freeWorkers;
};

# endif // __ASSIGNMENT_ALGORITHM_H__
