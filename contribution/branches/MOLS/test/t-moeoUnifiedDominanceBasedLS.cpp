#include <eo>
#include <moeo>
#include <moeoPopNeighborhoodExplorer.h>
#include <moeoPopLS.h>
#include <moeoUnifiedDominanceBasedLS.h>
#include <moMove.h>


class ObjectiveVectorTraits : public moeoObjectiveVectorTraits
{
public:
    static bool minimizing (int i)
    {
        return true;
    }
    static bool maximizing (int i)
    {
        return false;
    }
    static unsigned int nObjectives ()
    {
        return 2;
    }
};

typedef moeoRealObjectiveVector < ObjectiveVectorTraits > ObjectiveVector;

class Solution : public moeoRealVector < ObjectiveVector, double, double >
{
public:
    Solution() : moeoRealVector < ObjectiveVector, double, double > (1) {}
};

class dummyMove : public moMove < Solution >
{
public :
  void operator () (Solution & _solution){}
} ;


int main()
{

	// objective vectors
	    ObjectiveVector obj0, obj1, obj2, obj3, obj4, obj5, obj6;
	    obj0[0] = 2;
	    obj0[1] = 5;
	    obj1[0] = 3;
	    obj1[1] = 3;
	    obj2[0] = 4;
	    obj2[1] = 1;
	    obj3[0] = 5;
	    obj3[1] = 5;

	    // population
	    eoPop < Solution > pop;
	    pop.resize(4);
	    pop[0].objectiveVector(obj0);    // class 1
	    pop[1].objectiveVector(obj1);    // class 1
	    pop[2].objectiveVector(obj2);    // class 1
	    pop[3].objectiveVector(obj3);    // class 3

	    eoTimeContinue < Solution > continuator(5);

	    moeoUnifiedDominanceBasedLS < dummyMove > algo(continuator);

	    algo(pop);

	std::cout << "OK c'est bon" << std::endl;
	return EXIT_SUCCESS;
}
