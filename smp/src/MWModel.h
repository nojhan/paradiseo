/*
<MWModel.h>
Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2012

Alexandre Quemy

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

#ifndef SMP_MWMODEL_H_
#define SMP_MWMODEL_H_

#include <cassert>
#include <thread>

#include <scheduler.h>
#include <algoDispatching.h>
#include <policiesDispatching.h>


#include <eo>


namespace paradiseo
{
namespace smp
{

/** MWModel: Master / Worker Model for multicore computation

The MW Model wraps any algorithm in order to dispatch load between threads.

@see smp::Worker, smp::Thread
*/

template<template <class> class EOAlgo, class EOT, class Policy = LinearPolicy>
class MWModel : public EOAlgo<EOT>
{
public:
    /**
     * Constructor
     * @param workersNb the number of workers
     * @param args... list of parameters according to the constructor of your algorithm
     */
    template<class... Args>
    MWModel(unsigned workersNb, Args&... args);

    /**
     * Constructor
     * @param args... list of parameters according to the constructor of your algorithm
     */
    template<class... Args>
    MWModel(Args&... args);

    ~MWModel();
    
    /**
     * Apply an unary functor to the population
     * @param func unary functor
     * @param pop population
     */
    void apply(eoUF<EOT&, void>& func, eoPop<EOT>& pop);

    /**
     * Evaluate the population
     * @param pop population to evaluate
     */
    void evaluate(eoPop<EOT>& pop);
    
    /**
     * Run the algorithm on population
     * @param pop population to run the algorithm
     */
    void operator()(eoPop<EOT>& pop);
    
protected:
    /**
     * Specific algorithm for eoEasyEA
     */
    void operator()(eoPop<EOT>& pop, const eoEasyEA_tag&);
    
    /**
     * Specific algorithm for EasyPSO
     */
    void operator()(eoPop<EOT>& pop, const eoEasyPSO_tag&);
    
    /**
     * Specific algorithm for eoSyncEasyPSO
     */
    void operator()(eoPop<EOT>& pop, const eoSyncEasyPSO_tag&);
    
    /**
     * If we don't know the algorithm type
     */
    void operator()(eoPop<EOT>& pop,const error_tag&);
    
    std::vector<std::thread*> workers;
    Scheduler<EOT,Policy> scheduler;
};

#include <MWModel.cpp>
#include <MWAlgo/MWAlgo.h>

}

}

#endif
