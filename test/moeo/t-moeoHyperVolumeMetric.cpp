/*
* <t-moeoHyperVolumeMetric.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Lille-Nord Europe, 2006-2008
* (C) OPAC Team, LIFL, 2002-2008
*
* Arnaud Liefooghe
* Jeremie Humeau
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
// t-moeoHyperVolumeMetric.cpp
//-----------------------------------------------------------------------------

#include <paradiseo/eo.h>
#include <paradiseo/moeo.h>
#include <assert.h>

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

class ObjectiveVectorTraits2 : public moeoObjectiveVectorTraits
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

typedef moeoRealObjectiveVector < ObjectiveVectorTraits2 > ObjectiveVector2;

//-----------------------------------------------------------------------------

int main()
{
    std::cout << "[moeoHyperVolumeMetric] => \n";

    // objective vectors
    std::vector < ObjectiveVector > set1;

    //test normalisation
    set1.resize(4);

    //test case
    set1[0][0] = 1;
    set1[0][1] = 3;

    set1[1][0] = 5;
    set1[1][1] = 4;

    set1[2][0] = 4;
    set1[2][1] = 5;

    set1[3][0] = 2;
    set1[3][1] = 12;

    
    moeoHyperVolumeMetric < ObjectiveVector > metric(true, 1.1);
    
    std::vector < eoRealInterval > bounds;

    metric.setup(set1);
    bounds = metric.getBounds();

    std::cout << "\t>test normalization =>"; 
    assert(bounds[0].minimum()==1.0);
    assert(bounds[0].maximum()==5.0);
    assert(bounds[0].range()==4.0);

    assert(bounds[1].minimum()==3.0);
    assert(bounds[1].maximum()==12.0);
    assert(bounds[1].range()==9.0);
    std::cout << "OK\n"; 
    
    set1.resize(0);
    std::cout << "\t>test bad param in method setup =>";
    try{
    	metric.setup(set1);
    	return EXIT_FAILURE;
    }
    catch (char const* e){
    	std::cout << "Ok\n";
    }

    //test method dominates
    std::vector <double> a;
    std::vector <double> b;
    
    a.resize(3);
    b.resize(3);
    
    a[0]=10.0;
    a[1]=10.0;
    a[2]=2.0;
    
    b[0]= 3.0;
    b[1]= 10.0;
    b[2]= 4.0;
    
    std::cout << "\t>test method dominates =>";
    assert(metric.dominates(a, b, 2));
    assert(!metric.dominates(a, b, 3));
    std::cout << "Ok\n";
    
    //test method swap
    std::vector < std::vector <double> > front;
    front.resize(5);
    front[0]=a;
    front[1]=a;
    front[2]=a;
    front[3]=b;
    front[4]=a;

    std::cout << "\t>test method swap =>";
    metric.swap(front, 1, 3);
    assert(front.size()== 5);
    for(unsigned int i=0; i<5; i++){
    	if(i == 1){
		    assert(front[i][0]==3.0);
		    assert(front[i][1]==10.0);
		    assert(front[i][2]==4.0);
    	}
    	else{
		    assert(front[i][0]==10.0);
		    assert(front[i][1]==10.0);
		    assert(front[i][2]==2.0);
    	}
    }
    std::cout << "Ok\n";
 
    
    //test method filter_nondominated_set
    std::vector< double > c;
    std::vector< double > d;
    std::vector< double > e;
    
    c.resize(3);
    d.resize(3);
    e.resize(3);
    
    c[0]=11.0;
    c[1]=9.0;
    c[2]=5.0;
    
    d[0]=7.0;
    d[1]=7.0;
    d[2]=7.0;
    
    e[0]=9.0;
    e[1]=10.5;
    e[2]=14.0;
    
    front[0]=a;
    front[1]=b;
    front[2]=c;
    front[3]=d;
    front[4]=e;
    
    std::cout << "\t>test method filter_nondominated_set =>";
    unsigned int res = metric.filter_nondominated_set(front, 5, 2);
    
    assert(res == 3);
    
    assert(front[0] == a);
    assert(front[1]== e);
    assert(front[2]== c);
    assert(front[3]== d);
    assert(front[4]== b);
    
    std::cout << "Ok\n";
    
    //test method surface_unchanged_to
    std::cout << "\t>test method surface_unchanged_set =>";
    
    front[4]= a;
    front[0]= c;
    front[2]= b;
    
    double min = metric.surface_unchanged_to(front, 4, 2);
    
    assert(min == 4.0);
    
    try{
    	metric.surface_unchanged_to(front, 0, 2);
    	return EXIT_FAILURE;
    }
    catch (char const* e){
    	std::cout << "Ok\n";
    }
    
    // test method reduce_nondominated_set
    
    std::cout << "\t>test method reduce_nondominated_set=>";
    
    res=metric.reduce_nondominated_set(front, 3, 1, 9.5);
    assert(res==2);
    
    std::cout << "Ok\n";
    
    //test method calc_hypervolume
    std::cout << "\t>test method calc_hypervolume=>";
    
    a.resize(2);
    b.resize(2);
    c.resize(2);
    front.resize(3);
    
    a[0]=1;
    a[1]=3;
    b[0]=2;
    b[1]=2;
    c[0]=3;
    c[1]=1;
    
    front[0]=a;
    front[1]=b;
    front[2]=c;
    
    double hyp=metric.calc_hypervolume(front, 3, 2);
    assert (hyp==6.0);
    
    a.resize(3);
    b.resize(3);
    c.resize(3);
    
    a[2]=1;
    b[2]=2;
    c[2]=3;
    
    front[0]=c;
    front[1]=a;	
    front[2]=b;
    
    hyp=metric.calc_hypervolume(front, 3, 3);
    assert(hyp==14.0);
    
    std::cout << "Ok\n";
    
    //test de l'hyperVolume
    std::cout << "\t>test operator()=>\n";
    std::vector <ObjectiveVector2> solution;
    solution.resize(3);
    solution[0][0]=3.0;
    solution[0][1]=1.0;
    solution[0][2]=3.0;
    solution[1][0]=2.0;
    solution[1][1]=2.0;
    solution[1][2]=2.0;
    solution[2][0]=1.0;
    solution[2][1]=3.0;
    solution[2][2]=1.0;
    
    ObjectiveVector2 ref_point;
    ref_point.resize(3);
    ref_point[0]=4.0;
    ref_point[1]=4.0;
    ref_point[2]=4.0;
    
    std::cout << "\t\t-without normalization and ref_point =>";
    moeoHyperVolumeMetric < ObjectiveVector2 > metric2(false, ref_point);  
    hyp=metric2(solution);
    assert(hyp==14.0);
    std::cout << " Ok\n";
      
    std::cout << "\t\t-with normalization and ref_point =>";
    ref_point[0]=1.5;
    ref_point[1]=1.5;
    ref_point[2]=1.5;
    moeoHyperVolumeMetric < ObjectiveVector2 > metric3(true, ref_point);  
    hyp=metric3(solution);
    assert(hyp==1.75);
    std::cout << " Ok\n";
    
    std::cout << "\t\t-without normalization and a coefficent rho =>";
    hyp=0.0;
    moeoHyperVolumeMetric < ObjectiveVector2 > metric4(false, 2);  
    hyp=metric4(solution);
    assert(hyp==100.0);
    std::cout << " Ok\n";
    
    std::cout << "\t\t-with normalization and a coefficent rho =>";
    hyp=0.0;
    moeoHyperVolumeMetric < ObjectiveVector2 > metric5(true, 1.5);  
    hyp=metric5(solution);
    assert(hyp==1.75);
    std::cout << " Ok\n";
     
    return EXIT_SUCCESS;
}

//-----------------------------------------------------------------------------
