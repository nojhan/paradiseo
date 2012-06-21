# ifndef __ASSIGNMENT_ALGORITHM_H__
# define __ASSIGNMENT_ALGORITHM_H__

struct AssignmentAlgorithm
{
    virtual int get( ) = 0;
    virtual int size( ) = 0;
    virtual void confirm( int wrkRank ) = 0;
};

struct DynamicAssignmentAlgorithm : public AssignmentAlgorithm
{
    public:
        DynamicAssignmentAlgorithm( int offset, int size )
        {
            for( int i = 0; offset + i < size; ++i)
            {
                availableWrk.push_back( offset + i );
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

        int size()
        {
            return availableWrk.size();
        }

        void confirm( int rank )
        {
            availableWrk.push_back( rank );
        }

    protected:
        std::vector< int > availableWrk;
};



# endif // __ASSIGNMENT_ALGORITHM_H__
