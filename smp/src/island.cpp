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

template<template <class> class EOAlgo, class EOT, class bEOT>
template<class... Args>
paradiseo::smp::Island<EOAlgo,EOT,bEOT>::Island(std::function<EOT(bEOT&)> _convertFromBase, std::function<bEOT(EOT&)> _convertToBase, eoPop<EOT>& _pop, IntPolicy<EOT>& _intPolicy, MigPolicy<EOT>& _migPolicy, Args&... args) :
    // The PPExpander looks for the continuator in the parameters pack.
    // The private inheritance of ContWrapper wraps the continuator and add islandNotifier.
    ContWrapper<EOT, bEOT>(Loop<Args...>().template findValue<eoContinue<EOT>>(args...), this),
    // We inject the wrapped continuator by tag dispatching method during the algorithm construction.
    algo(EOAlgo<EOT>(wrap_pp<eoContinue<EOT>>(this->ck,args)...)),
    // With the PPE we look for the eval function in order to evaluate EOT to integrate
    eval(Loop<Args...>().template findValue<eoEvalFunc<EOT>>(args...)),
    pop(_pop),
    intPolicy(_intPolicy),
    migPolicy(_migPolicy),
    stopped(false),
    model(nullptr),
    convertFromBase(_convertFromBase),
    convertToBase(_convertToBase)
{
    // Check in compile time the inheritance thanks to type_trait.
    static_assert(std::is_base_of<eoAlgo<EOT>,EOAlgo<EOT>>::value, "Algorithm must inherit from eoAlgo<EOT>");
}

template<template <class> class EOAlgo, class EOT, class bEOT>
template<class... Args>
paradiseo::smp::Island<EOAlgo,EOT,bEOT>::Island(eoPop<EOT>& _pop, IntPolicy<EOT>& _intPolicy, MigPolicy<EOT>& _migPolicy, Args&... args) :
    Island(
    // Default conversion functions for homogeneous islands
    [](bEOT& i) -> EOT { return std::forward<EOT>(i); },
    [](EOT& i) -> bEOT { return std::forward<bEOT>(i); },
    _pop, _intPolicy, _migPolicy, args...)
{ }

template<template <class> class EOAlgo, class EOT, class bEOT>
void paradiseo::smp::Island<EOAlgo,EOT,bEOT>::operator()()
{
    stopped = false;
    algo(pop);
    stopped = true;
    // Let's wait the end of communications with the island model
    for(auto& message : sentMessages)
        message.join();
}

template<template <class> class EOAlgo, class EOT, class bEOT>
void paradiseo::smp::Island<EOAlgo,EOT,bEOT>::setModel(IslandModel<bEOT>* _model)
{
    model = _model;
}

template<template <class> class EOAlgo, class EOT, class bEOT>
eoPop<EOT>& paradiseo::smp::Island<EOAlgo,EOT,bEOT>::getPop() const
{
    return pop;
}

template<template <class> class EOAlgo, class EOT, class bEOT>
void paradiseo::smp::Island<EOAlgo,EOT,bEOT>::check()
{
    // Sending
    for(PolicyElement<EOT>& elem : migPolicy)
        if(!elem(pop))
            send(elem.getSelect());
    
    // Receiving
    receive();    
}

template<template <class> class EOAlgo, class EOT, class bEOT>
bool paradiseo::smp::Island<EOAlgo,EOT,bEOT>::isStopped(void) const
{
    return (bool)stopped;
}

template<template <class> class EOAlgo, class EOT, class bEOT>
void paradiseo::smp::Island<EOAlgo,EOT,bEOT>::send(eoSelect<EOT>& _select)
{
    // Allow island to work alone
    if(model != nullptr)
    {
        eoPop<EOT> migPop;
        _select(pop, migPop);

        // Convert pop to base pop
        eoPop<bEOT> baseMigPop;
        for(auto& indi : migPop)
            baseMigPop.push_back(convertToBase(indi));
            
        //std::cout << "On envoie de l'île : " << migPop << std::endl;
       
        // Delete delivered messages
        for(auto it = sentMessages.begin(); it != sentMessages.end(); it++)
            if(!it->joinable())
                sentMessages.erase(it);
      
        sentMessages.push_back(std::thread(&IslandModel<bEOT>::update, model, baseMigPop, this));
    }
}

template<template <class> class EOAlgo, class EOT, class bEOT>
void paradiseo::smp::Island<EOAlgo,EOT,bEOT>::receive(void)
{
    std::lock_guard<std::mutex> lock(this->m);
    while (!listImigrants.empty())
    { 
        //std::cout << "On reçoit dans l'île : " << listImigrants.size() << std::endl;
        eoPop<bEOT> base_offspring = listImigrants.front();
        
        // Convert objects from base to our objects type
        eoPop<EOT> offspring;
        for(auto& indi : base_offspring)
            offspring.push_back(convertFromBase(indi));
        
        // Evaluate objects to integrate
        for(auto& indi : offspring)
            eval(indi);
        
        intPolicy(pop, offspring);
        listImigrants.pop();

    }
}

template<template <class> class EOAlgo, class EOT, class bEOT>
void paradiseo::smp::Island<EOAlgo,EOT,bEOT>::update(eoPop<bEOT> _data)
{
    //std::cout << "On update dans l'île" << std::endl;
    std::lock_guard<std::mutex> lock(this->m);
    listImigrants.push(_data);
}

