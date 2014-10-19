//-----------------------------------------------------------------------------
// t-moeoQuickUnboundedArchive.cpp
//-----------------------------------------------------------------------------

#include <paradiseo/eo.h>
#include <paradiseo/moeo.h>

//-----------------------------------------------------------------------------

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

typedef MOEO < ObjectiveVector, double, double > Solution;

//-----------------------------------------------------------------------------

int main()
{
    std::cout << "[moeoArchive]\t=>\t";
    // First test is just to verify behavior of moeoQuickUnboundedArchive
    // objective vectors
    ObjectiveVector obj0, obj1, obj2, obj3, obj4, obj5,obj10;
    obj0[0] = 12226;
    obj0[1] = 427894;
    obj1[0] = 12170;
    obj1[1] = 431736;
    obj2[0] = 11965;
    obj2[1] = 435193;
    obj3[0] = 11893;
    obj3[1] = 441839;
    obj4[0] = 11870;
    obj4[1] = 450770;
    obj5[0] = 11769;
    obj5[1] = 460005;
    obj10[0] = 11769;
    obj10[1] = 46005;
    // population
    eoPop < Solution > pop;
    pop.resize(6);
    pop[0].objectiveVector(obj0);
    pop[1].objectiveVector(obj1);
    pop[2].objectiveVector(obj2);
    pop[3].objectiveVector(obj3);
    pop[4].objectiveVector(obj4);
    pop[5].objectiveVector(obj5);

    // archive
    moeoQuickUnboundedArchiveIndex< Solution> index;
    moeoIndexedArchive <Solution> arch(index);
//    moeoQuickUnboundedArchive< Solution > arch;
    arch(pop);
//    arch.printIndex();

    // size
    if (arch.size() > 6)
    {
        std::cout << "ERROR1 (too much solutions)" << std::endl;
        return EXIT_FAILURE;
    }
    // obj0 must be in
    if (! arch.contains(obj0))
    {
        std::cout << "ERROR2 (obj0 not in)" << std::endl;
        return EXIT_FAILURE;
    }
    // obj1 must be in
    if (! arch.contains(obj1))
    {
        std::cout << "ERROR3 (obj1 not in)" << std::endl;
        return EXIT_FAILURE;
    }
    // obj2 must be in
    if (! arch.contains(obj2))
    {
        std::cout << "ERROR4 (obj2 not in)" << std::endl;
        return EXIT_FAILURE;
    }
    // obj3 must be in
    if (! arch.contains(obj3))
    {
        std::cout << "ERROR5 (obj3 not in)" << std::endl;
        return EXIT_FAILURE;
    }
    // obj4 must be in
    if (!arch.contains(obj4))
    {
        std::cout << "ERROR6 (obj4 not in)! " << obj4<< std::endl;
//	arch.printIndex();
        return EXIT_FAILURE;
    }
    // obj5 must be in
    if (! arch.contains(obj5))
    {
        std::cout << "ERROR7 (obj5 not in)" << std::endl;
        return EXIT_FAILURE;
    }
    
    // Second test is to verify behavior with negative and comparator 
    ObjectiveVector obj6, obj7;
    std::cout<<"test neg"<<std::endl;
    obj6[0] = -12170;
    obj6[1] = 427894;
    obj7[0] = -12226;
    obj7[1] = 431736;
    eoPop < Solution > pop2;
    pop2.resize(2);
    pop2[0].objectiveVector(obj6);
    pop2[1].objectiveVector(obj7);
    std::cout<<"archive 2"<<std::endl;
    moeoQuickUnboundedArchiveIndex< Solution > index2;
    moeoIndexedArchive< Solution > arch2(index2);
    arch2(pop2);

    // size
    if (arch2.size() != 2)
    {
        std::cout << "ERROR8 (too much solutions)" << std::endl;
        return EXIT_FAILURE;
    }

    //Third test is with two equals values
    ObjectiveVector obj8, obj9;
    obj8[0] = 10;
    obj8[1] = 10;
    obj9[0] = 10;
    obj9[1] = 10;
    eoPop < Solution > pop3;
    pop3.resize(2);
    pop3[0].objectiveVector(obj8);
    pop3[1].objectiveVector(obj9);
    
    std::cout<<"archive 3"<<std::endl;
    moeoQuickUnboundedArchiveIndex< Solution > index3;
    moeoIndexedArchive< Solution > arch3(index3);
    arch3(pop3);

    if (arch3.size() != 1)
    {
        std::cout << "ERROR9 (too much solutions)" << std::endl;
        return EXIT_FAILURE;
    }
    eoPop < Solution > pop4;
    pop4.resize(6);
    pop4[0].objectiveVector(obj0);
    pop4[1].objectiveVector(obj1);
    pop4[2].objectiveVector(obj2);
    pop4[3].objectiveVector(obj3);
    pop4[4].objectiveVector(obj4);
    pop4[5].objectiveVector(obj10);
    std::cout<<"archive 4"<<std::endl;
    moeoQuickUnboundedArchiveIndex< Solution > index4;
    moeoIndexedArchive< Solution > arch4(index4);
    arch4(pop4);
    if (arch4.size() != 1)
    {
        std::cout << "ERROR10 (too much solutions)" << std::endl;
        return EXIT_FAILURE;
    }

    
    std::cout << "OK" << std::endl;
    return EXIT_SUCCESS;
}

