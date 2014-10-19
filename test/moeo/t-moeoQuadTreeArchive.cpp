	/*
* <t-moeoquadTreeArchive.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Arnaud Liefooghe
*
* This software is governed by the CeCILL license under French law and
* abiding by the rules of distribution of free software.  You can  use,
* modify and/ or redistribute the software under the terms of the CeCILL
* license as circulated by CEA, CNRS and INRIA at the following URL
* "http://www.cecill.info".
*
* As a counterpart to the access to the source code and  rights to copy,
* modify and redistribute granted by the license, users are provided only
* with a limited warranty  and the software's author,  the holder of the
* economic rights,  and the successive licensors  have only  limited liability.
*
* In this respect, the user's attention is drawn to the risks associated
* with loading,  using,  modifying and/or developing or reproducing the
* software by the user in light of its specific status of free software,
* that may mean  that it is complicated to manipulate,  and  that  also
* therefore means  that it is reserved for developers  and  experienced
* professionals having in-depth computer knowledge. Users are therefore
* encouraged to load and test the software's suitability as regards their
* requirements in conditions enabling the security of their systems and/or
* data to be ensured and,  more generally, to use and operate it in the
* same conditions as regards security.
* The fact that you are presently reading this means that you have had
* knowledge of the CeCILL license and that you accept its terms.
*
* ParadisEO WebSite : http://paradiseo.gforge.inria.fr
* Contact: paradiseo-help@lists.gforge.inria.fr
*
*/
//-----------------------------------------------------------------------------
// t-moeoEpsilonHyperboxArchive.cpp
//-----------------------------------------------------------------------------

#include <paradiseo/eo.h>
#include <paradiseo/moeo.h>
#include <cmath>

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
        return 3;
    }
};

typedef moeoRealObjectiveVector < ObjectiveVectorTraits > ObjectiveVector;

typedef MOEO < ObjectiveVector, double, double > Solution;





//-----------------------------------------------------------------------------

int main()
{
    std::cout << "[moeoQuadTreeArchive]\t=>\t";
    moeoQuadTree<ObjectiveVector> tree;

    bool empty= tree.isEmpty();
    std::cout <<"empty? " << empty << std::endl;
    ObjectiveVector obj1;
    obj1[0]=10.0;
    obj1[1]=10.0;
    obj1[2]=10.0;
    ObjectiveVector obj2;
    obj2[0]=9.0;
    obj2[1]=9.0;
    obj2[2]=9.0;
    ObjectiveVector obj3;
    obj3[0]=2.0;
    obj3[1]=11.0;
    obj3[2]=11.0;
    ObjectiveVector obj4;
    obj4[0]=1.0;
    obj4[1]=10.0;
    obj4[2]=10.0;
    ObjectiveVector obj5;
    obj5[0]=2.0;
    obj5[1]=2.0;
    obj5[2]=2.0;
    ObjectiveVector obj6;
    obj6[0]=26.0;
    obj6[1]=0.0;
    obj6[2]=5.0;
    ObjectiveVector obj7;
    obj7[0]=56.0;
    obj7[1]=22.0;
    obj7[2]=0.0;
    ObjectiveVector obj8;
    obj8[0]=87.0;
    obj8[1]=42.0;
    obj8[2]=62.0;
    ObjectiveVector obj9;
    obj9[0]=90.0;
    obj9[1]=69.0;
    obj9[2]=83.0;
    ObjectiveVector obj10;
    obj10[0]=68.0;
    obj10[1]=89.0;
    obj10[2]=22.0;
//    QuadTreeNode<ObjectiveVector> hop(obj1);
//    QuadTreeNode<ObjectiveVector> hop2(obj2);
//    QuadTreeNode<ObjectiveVector> hop3(obj3);
//    QuadTreeNode<ObjectiveVector> hop4(obj4);
//    empty = hop.getSubTree().empty();
//    std::cout <<"empty? " << empty << std::endl;
//    std::vector< QuadTreeNode<ObjectiveVector> > nodes;
//    nodes.push_back(hop);
//    nodes.push_back(hop2);
//    nodes.push_back(hop3);
//    std::cout << nodes[1].getVec() << std::endl;

//    std::cout << "size: " << nodes.size() << std::endl;
//    tree.insert(obj1);
//    tree.insert(obj2);
//    tree.insert(obj3);
//    tree.insert(obj4);
//    tree.insert(obj5);
    std::cout << "\n\n\n";

//    tree.insert(obj6);
//    tree.insert(obj7);
//    tree.insert(obj8);
//    tree.insert(obj9);
//    tree.insert(obj10);

    moeoUnboundedArchive<Solution> archive(false);
    eoPop<Solution> pop;
    pop.resize(1000);
    int tmp;

    for(int i= 0; i<1000 ; i++){
        ObjectiveVector obj;
    	obj[0]=floor(rng.uniform()*100);
    	obj[1]=floor(rng.uniform()*100);
    	obj[2]=floor(rng.uniform()*100);
    	std::cout << obj << std::endl;
    	pop[i].objectiveVector(obj);
    	tree.insert(obj);
    	archive(pop[i]);
    	tree.printTree();
        std::cout << std::endl;
        std::cout << std::endl;

        std::cout << "archive: " << archive << std::endl;
//        std::cin >> tmp;
    }



//    QuadTreeNode<ObjectiveVector> * a = tree.getRoot();
//    QuadTreeNode<ObjectiveVector> * b = a->getSubTree()[1];
//    QuadTreeNode<ObjectiveVector> * c = b->getSubTree()[2];
//
//    tree.reinsert(a,c);

//    std::cout << "achive: " << archive << std::endl;
    tree.printTree();




    std::cout << "OK" << std::endl;
    return EXIT_SUCCESS;
}

//-----------------------------------------------------------------------------
