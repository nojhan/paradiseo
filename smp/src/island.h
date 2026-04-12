/*
<island.h>
Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2012

Alexandre Quemy, Thibault Lasnier - INSA Rouen
Eremey Valetov

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

#ifndef SMP_ISLAND_H_
#define SMP_ISLAND_H_

#include <queue>
#include <vector>
#include <utility>
#include <atomic>
#include <future>
#include <type_traits>
#include <algorithm>

#include <eoEvalFunc.h>
#include <eoSelect.h>
#include <eoAlgo.h>
#include <eoPop.h>
#include <algo/moeoAlgoFinalized.h>

#include <abstractIsland.h>
#include <islandModel.h>
#include <migPolicy.h>
#include <intPolicy.h>
#include <PPExpander.h>
#include <contWrapper.h>
#include <contDispatching.h>

#include <MPI_IslandModel.h>

namespace paradiseo
{
namespace smp
{

enum IslandType {
    HOMOGENEOUS_ISLAND,
    HETEROGENEOUS_ISLAND,
    DEFAULT = HETEROGENEOUS_ISLAND
};

/** Island: Concrete island that wraps an algorithm

The island wraps an algorithm and provide mecanisms for emigration and integration of populations.
An island also have a base type which represents the type of individuals of the Island Model.
The optional algoEOT template parameter allows the algorithm to operate on a different
type than the island's EOT (useful when the algorithm's MOEOT differs from the base type).

@see smp::AbstractIsland, smp::MigPolicy
*/

template<template <class> class EOAlgo, class EOT, class bEOT = EOT, class algoEOT = EOT>
class Island : private ContWrapper<EOT, bEOT>, public AIsland<bEOT>
{
public:
    /**
     * Constructor with type converters
     * @param _convertFromBase Function to convert EOT from base EOT
     * @param _convertToBase Function to convert base EOT to EOT
     * @param _pop Population of the island
     * @param _intPolicy Integration policy
     * @param _migPolicy Migration policy
     * @param args Parameters to construct the algorithm.
     */
    template<class... Args>
    Island(std::function<EOT(bEOT&)> _convertFromBase, std::function<bEOT(EOT&)> _convertToBase,
           eoPop<EOT>& pop, IntPolicy<EOT>& _intPolicy, MigPolicy<EOT>& _migPolicy, Args&... args);

    /**
     * Constructor for homogeneous islands (bEOT == EOT)
     */
    template<class... Args>
    Island(eoPop<EOT>& pop, IntPolicy<EOT>& _intPolicy, MigPolicy<EOT>& _migPolicy, Args&... args);

    /**
     * Constructor with type converters and island type
     */
    template<class... Args>
    Island(std::function<EOT(bEOT&)> _convertFromBase, std::function<bEOT(EOT&)> _convertToBase,
           eoPop<EOT>& pop, IntPolicy<EOT>& _intPolicy, MigPolicy<EOT>& _migPolicy,
           IslandType islandType, Args&... args);

    /**
     * Constructor with island type
     */
    template<class... Args>
    Island(eoPop<EOT>& pop, IntPolicy<EOT>& _intPolicy, MigPolicy<EOT>& _migPolicy,
           IslandType islandType, Args&... args);

    void operator()(void);

    virtual void setModel(IslandModel<bEOT>* _model);
    virtual void setModel(MPI_IslandModel<bEOT>* _model);
    virtual void setModel(Redis_IslandModel<bEOT>* _model);

    eoPop<EOT>& getPop() const;
    virtual void check(void);
    bool update(eoPop<bEOT> _data);
    virtual bool isStopped(void) const;
    virtual void setRunning(void);
    virtual void receive(void);

protected:
    virtual void send(eoSelect<EOT>& _select);

    EOAlgo<algoEOT> algo;
    eoEvalFunc<EOT>& eval;
    eoPop<EOT>& pop;
    std::queue<eoPop<bEOT>> listImigrants;
    IntPolicy<EOT>& intPolicy;
    MigPolicy<EOT>& migPolicy;
    std::atomic<bool> stopped;
    std::vector<std::shared_future<bool>> sentMessages;

    enum class IslandModelKind { None, Shared, MPI, Redis };
    IslandModelKind modelKind_ = IslandModelKind::None;
    IslandModel<bEOT>* sharedModel_ = nullptr;
    MPI_IslandModel<bEOT>* mpiModel_ = nullptr;
    Redis_IslandModel<bEOT>* redisModel_ = nullptr;

    std::function<EOT(bEOT&)> convertFromBase;
    std::function<bEOT(EOT&)> convertToBase;
    IslandType _islandType = DEFAULT;
};

#include <island.cpp>

}

}

#endif
