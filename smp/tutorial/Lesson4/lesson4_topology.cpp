/*
<lesson4_topology.cpp>
Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2012

Alexandre Quemy, Thibault Lasnier - INSA Rouen

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software.  You can  ue,
modify and/ or redistribute the software under the terms of the CeCILL
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info".

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or
data to be ensured and,  more generally, to use and operate it in the
same conditions as regards security.
The fact that you are presently reading this means that you have had
knowledge of the CeCILL license and that you accept its terms.

ParadisEO WebSite : http://paradiseo.gforge.inria.fr
Contact: paradiseo-help@lists.gforge.inria.fr
*/

/*///////////////////////////////////////////////////////////////////
// SMP Tutorial 4 : Advanced Island Model Mecanisms
// This file shows how to create events called by islands.
// In our case, we would like to change the topology after 10 seconds
// of computation.
/*///////////////////////////////////////////////////////////////////

#include <smp>
#include <eo>
#include <chrono>

#include "../BaseLesson.h"

using namespace paradiseo::smp;
using namespace std;

/*
    Simple function to change topology of a given model, once, after 10 seconds.
*/
void changeTopo(IslandModel<Indi>* _model, AbstractTopology& _topo)
{
    static auto start = std::chrono::system_clock::now();
    auto now = std::chrono::system_clock::now();
    auto elapsed = now - start;
    static bool first = false;
    if(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() > 10000 && !first)
    {
        std::cout << "Changing topology !" << std::endl;
        _model->setTopology(_topo);
        first = true;
    }    
}

int main(void)
{
    // Defining parameters
    typedef struct {
        unsigned popSize = 1000;
        unsigned tSize = 2;
        double pCross = 0.8;
        double pMut = 0.7;
        unsigned maxGen = 1000;
    } Param; 

    Param param;

    // Load instance
    loadInstances("../lessonData.dat", n, bkv, a, b);
      
    //Common part to all islands
    IndiEvalFunc plainEval;
    IndiInit chromInit;
    eoDetTournamentSelect<Indi> selectOne(param.tSize);
    eoSelectPerc<Indi> select(selectOne);// by default rate==1
    IndiXover Xover;                 // CROSSOVER
    IndiSwapMutation  mutationSwap;  // MUTATION
    eoSGATransform<Indi> transform(Xover, param.pCross, mutationSwap, param.pMut);
    eoPlusReplacement<Indi> replace;
    
    // MODEL
    // Topologies
    Topology<Complete> topo;
    IslandModel<Indi> model(topo);
    
    // ISLAND 1
    // // Algorithm part
    eoGenContinue<Indi> genCont(param.maxGen+100);
    eoPop<Indi> pop(param.popSize, chromInit);
    // // Emigration policy
    // // // Element 1
    eoPeriodicContinue<Indi> criteria(5);
    eoDetTournamentSelect<Indi> selectOne1(20);
    eoSelectNumber<Indi> who(selectOne1, 3);
    
    Topology<Ring> topo2;
     
    /*
        We need to bind our function in a std::function object.
        Then, we create a Notifier that we add to our island thanks
        to an eoCheckPoint.
    */
    auto task = std::bind(changeTopo, &model, topo2);
    Notifier topoChanger(task);
    eoCheckPoint<Indi> ck(genCont);
    ck.add(topoChanger);
    
    MigPolicy<Indi> migPolicy;
    migPolicy.push_back(PolicyElement<Indi>(who, criteria));
        
    // // Integration policy
    eoPlusReplacement<Indi> intPolicy;

    Island<eoEasyEA,Indi> test(pop, intPolicy, migPolicy, ck, plainEval, select, transform, replace);

    try
    {
        
        model.add(test);
        
        // The topology will change after 10 seconds of computation
        model();
        
    }
    catch(exception& e)
    {
      cout << "Exception: " << e.what() << '\n';
    }
    
    return 0;
}
