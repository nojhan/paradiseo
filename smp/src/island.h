/*
<island.h>
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

#ifndef SMP_ISLAND_H_
#define SMP_ISLAND_H_

#include <queue>
#include <vector>
#include <utility>
#include <atomic>
#include <type_traits>

#include <eoEvalFunc.h>
#include <eoSelect.h>
#include <eoAlgo.h>
#include <eoPop.h>

#include <abstractIsland.h>
#include <islandModel.h>
#include <migPolicy.h>
#include <intPolicy.h>
#include <PPExpander.h>
#include <contWrapper.h>
#include <contDispatching.h>

namespace paradiseo
{
namespace smp
{
/** Island: Concrete island that wraps an algorithm

The island wraps an algorithm and provide mecanisms for emigration and integration of populations.
An island also have a base type which represents the type of individuals of the Island Model.

@see smp::AbstractIsland, smp::MigPolicy
*/

template<template <class> class EOAlgo, class EOT, class bEOT = EOT>
class Island : private ContWrapper<EOT, bEOT>, public AIsland<bEOT>
{
public:
    /**
     * Constructor
     * @param _convertFromBase Function to convert EOT from base EOT
     * @param _convertToBase Function to convert base EOT to EOT
     * @param _pop Population of the island
     * @param _intPolicy Integration policy
     * @param _migPolicy Migration policy
     * @param args Parameters to construct the algorithm.
     */
    template<class... Args>
    Island(std::function<EOT(bEOT&)> _convertFromBase, std::function<bEOT(EOT&)> _convertToBase, eoPop<EOT>& pop, IntPolicy<EOT>& _intPolicy, MigPolicy<EOT>& _migPolicy, Args&... args);
    /**
     * Constructor
     * @param _pop Population of the island
     * @param _intPolicy Integration policy
     * @param _migPolicy Migration policy
     * @param args Parameters to construct the algorithm.
     */
    template<class... Args>
    Island(eoPop<EOT>& pop, IntPolicy<EOT>& _intPolicy, MigPolicy<EOT>& _migPolicy, Args&... args);
    
    /**
     * Start the island.
     */
    void operator()(void);
    
    /**
     * Set model
     * @param _model Pointer to the Island Model corresponding 
     */
    virtual void setModel(IslandModel<bEOT>* _model);
    
    /**
     * Return a reference to the island population.
     * @return Reference to the island population
     */
    eoPop<EOT>& getPop() const;
    
    /**
     * Check if there is population to receive or to migrate
     */
    virtual void check(void);
    
    /**
     * Update the list of imigrants.
     * @param _data Elements to integrate in the main population.
     */
    void update(eoPop<bEOT> _data);
    
    /**
     * Check if the algorithm is stopped.
     * @return true if stopped
     */
    virtual bool isStopped(void) const; 
    
    /**
     * Check if there is population to receive
     */
    virtual void receive(void);
    
    AIsland<bEOT> clone() const;
    
protected:

    /**
     * Send population to mediator
     * @param _select Method to select EOT to send
     */
    virtual void send(eoSelect<EOT>& _select);
    
    eoEvalFunc<EOT>& eval;               
    eoPop<EOT>& pop;
    EOAlgo<EOT> algo;
    std::queue<eoPop<bEOT>> listImigrants;
    IntPolicy<EOT>& intPolicy;
    MigPolicy<EOT>& migPolicy;
    std::atomic<bool> stopped;
    std::vector<std::thread> sentMessages;
    IslandModel<bEOT>* model;
    std::function<EOT(bEOT&)> convertFromBase; 
    std::function<bEOT(EOT&)> convertToBase;
};

#include <island.cpp>

}

}

#endif
