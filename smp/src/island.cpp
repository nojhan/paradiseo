/*
<island.cpp>
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

#include <type_traits>


template<template <class> class EOAlgo, class EOT>
template<class... Args>
paradiseo::smp::Island<EOAlgo,EOT>::Island(unsigned _popSize, eoInit<EOT>& _chromInit, IntPolicy<EOT>& _intPolicy, MigPolicy<EOT>& _migPolicy, Args&... args) :
    // The PPExpander looks for the continuator in the parameters pack.
    // The private inheritance of ContWrapper wraps the continuator and add islandNotifier.
    ContWrapper<EOT>(Loop<Args...>().template findValue<eoContinue<EOT>>(args...), this),
    // We inject the wrapped continuator by tag dispatching method during the algorithm construction.
    algo(EOAlgo<EOT>(wrap_pp<eoContinue<EOT>>(this->ck,args)...)),
    // With the PPE we look for the eval function in order to evaluate EOT to integrate
    eval(Loop<Args...>().template findValue<eoEvalFunc<EOT>>(args...)),
    pop(_popSize, _chromInit),
    intPolicy(_intPolicy),
    migPolicy(_migPolicy),
    stopped(false)
{
    // Check in compile time the inheritance thanks to type_trait.
    static_assert(std::is_base_of<eoAlgo<EOT>,EOAlgo<EOT>>::value, "Algorithm must inherit from eoAlgo<EOT>");
}

template<template <class> class EOAlgo, class EOT>
void paradiseo::smp::Island<EOAlgo,EOT>::operator()()
{
    stopped = false;
    algo(pop);
    // Let's wait the end of communications with the island model
    for(auto& message : sentMessages)
        message.join();
    stopped = true;
}

template<template <class> class EOAlgo, class EOT>
void paradiseo::smp::Island<EOAlgo,EOT>::setModel(IslandModel<EOT>* _model)
{
    model = _model;
}

template<template <class> class EOAlgo, class EOT>
void paradiseo::smp::Island<EOAlgo,EOT>::update(eoPop<EOT>& _data)
{
    listImigrants.push_back(&_data);
}

template<template <class> class EOAlgo, class EOT>
eoPop<EOT>& paradiseo::smp::Island<EOAlgo,EOT>::getPop()
{
    return pop;
}

template<template <class> class EOAlgo, class EOT>
void paradiseo::smp::Island<EOAlgo,EOT>::check()
{
    // std::cout << "On check" << std::endl;
    for(PolicyElement<EOT>& elem : migPolicy)
    {
        if(!elem(pop))
        {
            // std::cout << "On lance l'emmigration" << std::endl;
            send(elem.getSelect());
        }
    }
    receive();    
}

template<template <class> class EOAlgo, class EOT>
bool paradiseo::smp::Island<EOAlgo,EOT>::isStopped(void)
{
    return (bool)stopped;
}

template<template <class> class EOAlgo, class EOT>
void paradiseo::smp::Island<EOAlgo,EOT>::send(eoSelect<EOT>& _select)
{
    //std::cout << "Ile lance la migration" << std::endl;
    eoPop<EOT> migPop;
    _select(pop, migPop);
    //std::cout << "   La population migrante est :" << std::endl << migPop << std::endl;
    
    // Delete delivered messages
    for(auto it = sentMessages.begin(); it != sentMessages.end(); it++)
        if(!it->joinable())
            sentMessages.erase(it);
  
    sentMessages.push_back(std::thread(&IslandModel<EOT>::update, model, migPop, this));

}

template<template <class> class EOAlgo, class EOT>
void paradiseo::smp::Island<EOAlgo,EOT>::receive(void)
{
    while (!listImigrants.empty())
    {
        eoPop<EOT> offspring = *(listImigrants.front());
        
        // Evaluate objects to integrate
        for(auto& indi : offspring)
            eval(indi);
        
        intPolicy(pop, offspring);
        listImigrants.pop();
        
    }
}
