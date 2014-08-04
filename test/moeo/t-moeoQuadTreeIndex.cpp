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

class ObjectiveVectorTraits3d : public moeoObjectiveVectorTraits
{
public:
    static bool minimizing (int i)
    {
        return true ;
    }
    static bool maximizing (int i)
    {
        return false ;
    }
    static unsigned int nObjectives ()
    {
        return 3;
    }
};


typedef moeoRealObjectiveVector < ObjectiveVectorTraits > ObjectiveVector;
typedef moeoRealObjectiveVector < ObjectiveVectorTraits3d > ObjectiveVector3d;

typedef MOEO < ObjectiveVector, double, double > Solution;
typedef MOEO < ObjectiveVector3d, double, double > Solution3d;

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
    moeoQuadTree<Solution> index;
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
    obj6[0] = -12170;
    obj6[1] = 427894;
    obj7[0] = -12226;
    obj7[1] = 431736;
    eoPop < Solution > pop2;
    pop2.resize(2);
    pop2[0].objectiveVector(obj6);
    pop2[1].objectiveVector(obj7);
    moeoQuadTree< Solution > index2;
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

    moeoQuadTree< Solution > index3;
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
    moeoQuadTree< Solution > index4;
    moeoIndexedArchive< Solution > arch4(index4);
    arch4(pop4);
    if (arch4.size() != 1)
    {
        std::cout << "ERROR10 (too much solutions)" << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<bool>bobj;
    for (unsigned int i=0;i<3;i++){
	    bobj.push_back(false);
    }

    ObjectiveVectorTraits3d::setup(3,bobj);
    moeoQuadTree < Solution3d> index5;
    moeoIndexedArchive < Solution3d> arch5(index5);
    eoPop<Solution3d> pop5;
    pop5.resize(6);
    ObjectiveVector3d obj3d0, obj3d1, obj3d2, obj3d3, obj3d4, obj3d5, obj3d6,obj3d10;
    obj3d0[0] = 12226;
    obj3d0[1] = 427894;
    obj3d0[2] = 10;

    obj3d1[0] = 12170;
    obj3d1[1] = 431736;
    obj3d1[2] = 10;

    obj3d2[0] = 11965;
    obj3d2[1] = 435193;
    obj3d2[2] = 10;

    obj3d3[0] = 11893;
    obj3d3[1] = 441839;
    obj3d3[2] = 10;

    obj3d4[0] = 11870;
    obj3d4[1] = 450770;
    obj3d4[2] = 10;

    obj3d5[0] = 11769;
    obj3d5[1] = 46005;
    obj3d5[2] = 2;

    obj3d10[0] = 11769;
    obj3d10[1] = 46005;
    obj3d10[2] = 10;

    obj3d6[0] = 11769;
    obj3d6[1] = 460005;
    obj3d6[2] = 1;


    pop5[0].objectiveVector(obj3d0);
    pop5[1].objectiveVector(obj3d1);
    pop5[2].objectiveVector(obj3d2);
    pop5[3].objectiveVector(obj3d3);
    pop5[4].objectiveVector(obj3d4);
    pop5[5].objectiveVector(obj3d10);



    arch5(pop5);
    pop5.resize(7);
    pop5[6].objectiveVector(obj3d5);
    arch5(pop5[6]);
//    index5.printTree();
/*    for (unsigned int i=0;i<arch5.size();i++){
	    std::cout<<i<<" "<<arch5[i].objectiveVector()<<std::endl;
    }*/
    assert(arch5.size()==1);
    pop5.resize(8);
    pop5[7].objectiveVector(obj3d6);
    arch5(pop5[7]);
    assert(arch5.size()==2);

    moeoQuadTree < Solution3d> index6;
    moeoIndexedArchive < Solution3d> arch6(index6);
    eoPop<Solution3d> pop6;

    ObjectiveVector3d jojo, jojo1, jojo2, jojo3, jojo4, jojo5, jojo6, jojo7, jojo8, jojo9 ;
    jojo[0]=10;
    jojo[1]=10;
    jojo[2]=10;

    jojo1[0]=5;
    jojo1[1]=5;
    jojo1[2]=23;

    jojo2[0]=3;
    jojo2[1]=25;
    jojo2[2]=16;

    jojo3[0]=14;
    jojo3[1]=18;
    jojo3[2]=6;

    jojo4[0]=100;
    jojo4[1]=100;
    jojo4[2]=100;

    jojo5[0]=4;
    jojo5[1]=8;
    jojo5[2]=12;


    jojo6[0]=11;
    jojo6[1]=14;
    jojo6[2]=9;


    jojo7[0]=40;
    jojo7[1]=12;
    jojo7[2]=3;

    jojo8[0]=12;
    jojo8[1]=8;
    jojo8[2]=4;

    jojo9[0]=4;
    jojo9[1]=4;
    jojo9[2]=22;

    pop6.resize(8);
    pop6[0].objectiveVector(jojo);
    pop6[1].objectiveVector(jojo1);
    pop6[2].objectiveVector(jojo2);
    pop6[3].objectiveVector(jojo3);
    pop6[4].objectiveVector(jojo4);
    pop6[5].objectiveVector(jojo5);
    pop6[6].objectiveVector(jojo6);
    pop6[7].objectiveVector(jojo7);
    Solution3d add;
    Solution3d add2;
    add.objectiveVector(jojo8);
    add2.objectiveVector(jojo9);
   /* pop[1].objectiveVector(jojo1);
    pop[1].objectiveVector(jojo1);
    pop[1].objectiveVector(jojo1);*/
    arch6(pop6);

    assert(arch6.size()==7);
    bool res=arch6(add);
    assert(res && arch6.size()==7);
    res=arch6(add2);
    assert(res && arch6.size()==7);
    std::cout << "OK" << std::endl;
    return EXIT_SUCCESS;
}

