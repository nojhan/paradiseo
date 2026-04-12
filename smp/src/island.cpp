/*
<island.cpp>
Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2012

Alexandre Quemy, Thibault Lasnier - INSA Rouen
Eremey Valetov

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software.  You can  ue,
modify and/ or redistribute the software under the terms of the CeCILL
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info".

ParadisEO WebSite : http://paradiseo.gforge.inria.fr
Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifdef _OPENMP
#include <omp.h>
#endif
#include <utils/eoLogger.h>

template<template <class> class EOAlgo, class EOT, class bEOT, class algoEOT>
template<class... Args>
paradiseo::smp::Island<EOAlgo,EOT,bEOT,algoEOT>::Island(
    std::function<EOT(bEOT&)> _convertFromBase, std::function<bEOT(EOT&)> _convertToBase,
    eoPop<EOT>& _pop, IntPolicy<EOT>& _intPolicy, MigPolicy<EOT>& _migPolicy, Args&... args) :
    ContWrapper<EOT, bEOT>(Loop<Args...>().template findValue<eoContinue<EOT>>(args...), this),
    algo(EOAlgo<algoEOT>(wrap_pp<eoContinue<EOT>>(this->ck,args)...)),
    eval(Loop<Args...>().template findValue<eoEvalFunc<EOT>>(args...)),
    pop(_pop),
    intPolicy(_intPolicy),
    migPolicy(_migPolicy),
    stopped(false),
    convertFromBase(_convertFromBase),
    convertToBase(_convertToBase)
{}

template<template <class> class EOAlgo, class EOT, class bEOT, class algoEOT>
template<class... Args>
paradiseo::smp::Island<EOAlgo,EOT,bEOT,algoEOT>::Island(
    eoPop<EOT>& _pop, IntPolicy<EOT>& _intPolicy, MigPolicy<EOT>& _migPolicy, Args&... args) :
    Island(
    [](bEOT& i) -> EOT { return std::forward<EOT>(i); },
    [](EOT& i) -> bEOT { return std::forward<bEOT>(i); },
    _pop, _intPolicy, _migPolicy, args...)
{}

template<template <class> class EOAlgo, class EOT, class bEOT, class algoEOT>
template<class... Args>
paradiseo::smp::Island<EOAlgo,EOT,bEOT,algoEOT>::Island(
    std::function<EOT(bEOT&)> _convertFromBase, std::function<bEOT(EOT&)> _convertToBase,
    eoPop<EOT>& _pop, IntPolicy<EOT>& _intPolicy, MigPolicy<EOT>& _migPolicy,
    IslandType islandType, Args&... args) :
    ContWrapper<EOT, bEOT>(Loop<Args...>().template findValue<eoContinue<EOT>>(args...), this),
    algo(EOAlgo<algoEOT>(wrap_pp<eoContinue<EOT>>(this->ck,args)...)),
    eval(Loop<Args...>().template findValue<eoEvalFunc<EOT>>(args...)),
    pop(_pop),
    intPolicy(_intPolicy),
    migPolicy(_migPolicy),
    stopped(false),
    convertFromBase(_convertFromBase),
    convertToBase(_convertToBase),
    _islandType(islandType)
{}

template<template <class> class EOAlgo, class EOT, class bEOT, class algoEOT>
template<class... Args>
paradiseo::smp::Island<EOAlgo,EOT,bEOT,algoEOT>::Island(
    eoPop<EOT>& _pop, IntPolicy<EOT>& _intPolicy, MigPolicy<EOT>& _migPolicy,
    IslandType islandType, Args&... args) :
    Island(
    [](bEOT& i) -> EOT { return std::forward<EOT>(i); },
    [](EOT& i) -> bEOT { return std::forward<bEOT>(i); },
    _pop, _intPolicy, _migPolicy, islandType, args...)
{}

template<template <class> class EOAlgo, class EOT, class bEOT, class algoEOT>
void paradiseo::smp::Island<EOAlgo,EOT,bEOT,algoEOT>::operator()()
{
    stopped = false;
    algo(pop);
    stopped = true;
    for (auto& message : sentMessages)
        message.wait();
    sentMessages.clear();
}

template<template <class> class EOAlgo, class EOT, class bEOT, class algoEOT>
void paradiseo::smp::Island<EOAlgo,EOT,bEOT,algoEOT>::setModel(IslandModel<bEOT>* _model)
{
    sharedModel_ = _model;
    modelKind_ = IslandModelKind::Shared;
}

template<template <class> class EOAlgo, class EOT, class bEOT, class algoEOT>
void paradiseo::smp::Island<EOAlgo,EOT,bEOT,algoEOT>::setModel(MPI_IslandModel<bEOT>* _model)
{
    mpiModel_ = _model;
    modelKind_ = IslandModelKind::MPI;
}

template<template <class> class EOAlgo, class EOT, class bEOT, class algoEOT>
void paradiseo::smp::Island<EOAlgo,EOT,bEOT,algoEOT>::setModel(Redis_IslandModel<bEOT>* _model)
{
    redisModel_ = _model;
    modelKind_ = IslandModelKind::Redis;
}

template<template <class> class EOAlgo, class EOT, class bEOT, class algoEOT>
eoPop<EOT>& paradiseo::smp::Island<EOAlgo,EOT,bEOT,algoEOT>::getPop() const
{
    return pop;
}

template<template <class> class EOAlgo, class EOT, class bEOT, class algoEOT>
void paradiseo::smp::Island<EOAlgo,EOT,bEOT,algoEOT>::check()
{
    for (PolicyElement<EOT>& elem : migPolicy)
        if (!elem(pop)) {
            eo::log << eo::debug << "Island::check: migration policy triggered, sending migrants (pop size=" << pop.size() << ")" << std::endl;
            send(elem.getSelect());
        }
    receive();
}

template<template <class> class EOAlgo, class EOT, class bEOT, class algoEOT>
bool paradiseo::smp::Island<EOAlgo,EOT,bEOT,algoEOT>::isStopped(void) const
{
    return (bool)stopped;
}

template<template <class> class EOAlgo, class EOT, class bEOT, class algoEOT>
void paradiseo::smp::Island<EOAlgo,EOT,bEOT,algoEOT>::setRunning(void)
{
    stopped = false;
}

template<template <class> class EOAlgo, class EOT, class bEOT, class algoEOT>
void paradiseo::smp::Island<EOAlgo,EOT,bEOT,algoEOT>::send(eoSelect<EOT>& _select)
{
    if (modelKind_ == IslandModelKind::None)
        return;

    eoPop<EOT> migPop;
    _select(pop, migPop);
    eo::log << eo::debug << "Island::send: selected " << migPop.size() << " migrant(s) from pop of " << pop.size() << std::endl;

    eoPop<bEOT> baseMigPop;
    for (auto& indi : migPop)
        baseMigPop.push_back(std::move(convertToBase(indi)));

    sentMessages.erase(std::remove_if(sentMessages.begin(), sentMessages.end(),
        [](std::shared_future<bool>& i) -> bool
        { return i.wait_for(std::chrono::nanoseconds(0)) == std::future_status::ready; }
        ),
        sentMessages.end());

    switch (modelKind_) {
        case IslandModelKind::Shared:
            sentMessages.push_back(std::async(std::launch::async,
                &IslandModel<bEOT>::update, sharedModel_,
                std::move(baseMigPop), this));
            break;
        case IslandModelKind::MPI:
            sentMessages.push_back(std::async(std::launch::async,
                &MPI_IslandModel<bEOT>::update, mpiModel_,
                std::move(baseMigPop), this));
            break;
        case IslandModelKind::Redis:
            sentMessages.push_back(std::async(std::launch::async,
                &Redis_IslandModel<bEOT>::update, redisModel_,
                std::move(baseMigPop), this));
            break;
        case IslandModelKind::None:
            break;
    }
}

template<template <class> class EOAlgo, class EOT, class bEOT, class algoEOT>
void paradiseo::smp::Island<EOAlgo,EOT,bEOT,algoEOT>::receive(void)
{
    std::lock_guard<std::mutex> lock(this->m);

    while (!listImigrants.empty()) {
        eoPop<bEOT> base_offspring = std::move(listImigrants.front());
        eo::log << eo::debug << "Island::receive: integrating " << base_offspring.size()
                << " immigrant(s) into pop of " << pop.size() << std::endl;

        eoPop<EOT> offspring;
        for (auto& indi : base_offspring)
            offspring.push_back(std::move(convertFromBase(indi)));

        // Re-evaluate immigrants for heterogeneous islands
        if (_islandType == HETEROGENEOUS_ISLAND) {
#ifdef _OPENMP
            #pragma omp parallel for
#endif
            for (size_t i = 0; i < offspring.size(); ++i) {
                offspring[i].invalidate();
                eval(offspring[i]);
            }
        }

        intPolicy(pop, offspring);

        if constexpr (std::is_base_of_v<moeoAlgoFinalized<algoEOT>, EOAlgo<algoEOT>>)
            algo.finalize(pop);

        listImigrants.pop();
    }
}

template<template <class> class EOAlgo, class EOT, class bEOT, class algoEOT>
bool paradiseo::smp::Island<EOAlgo,EOT,bEOT,algoEOT>::update(eoPop<bEOT> _data)
{
    std::lock_guard<std::mutex> lock(this->m);
    listImigrants.push(_data);
    return true;
}
