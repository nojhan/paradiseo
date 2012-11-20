/*
<scheduler.h>
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

#ifndef SCHEDULER_H_
#define SCHEDULER_H_

#include <iostream>
#include <vector>
#include <atomic>
#include <mutex>

#include <thread.h>
#include <policiesDispatching.h>

#include <eoEvalFunc.h>
#include <eoPop.h>


namespace paradiseo
{
namespace smp
{
/** Scheduler : Dispatch load between workers according to a policy.

Dispatch load between the specified number of workers according to a policy.

*/

template<class EOT, class Policy = LinearPolicy>
class Scheduler
{
public:
    /**
     * Constructor
     * @param workersNb number of workers to perform tasks
     */
    Scheduler(unsigned workersNb);
    
    /**
     * Destructor
     */
    ~Scheduler();
    
    /**
     * Start an unary functor on workers on all the pop
     * @param func unary functor
     * @param pop reference to the population
     */
    void operator()(eoUF<EOT&, void>& func, eoPop<EOT>& pop);

protected:
    /**
     * Perform scheduling with a linear policy
     */
    void operator()(eoUF<EOT&, void>& func, eoPop<EOT>& pop, const LinearPolicy&);
    
    /**
     * Perform scheduling with a linear policy
     */
    void operator()(eoUF<EOT&, void>& func, eoPop<EOT>& pop, const ProgressivePolicy&);

    /**
     * Apply an unary functor on a sub-group of population
     * @param func unary functor
     * @param pop reference to the sub-group
     */
    void applyLinearPolicy(eoUF<EOT&, void>& func, std::vector<EOT*>& pop);
    
    /**
     * Apply an unary functor on a sub-group of population
     * @param func unary functor
     * @param pop reference to the sub-group
     * @param id id of the thread
     */
    void applyProgressivePolicy(eoUF<EOT&, void>& func, std::vector<EOT*>& pop, int id);
    
    /**
     * Create sub-groups with similar size from a population.
     * @param pop reference to the pop
     */
    std::vector<std::vector<EOT*>> subGroups(eoPop<EOT>& pop);
    
    std::vector<Thread> workers;
    std::vector<std::vector<EOT*>> popPackages;
    
    std::atomic<bool> done;
    std::vector<unsigned> planning;
    std::vector<std::atomic<int>> isWorking;
    std::vector<std::mutex> m;
};

#include <scheduler.cpp>

}

}

#endif /*SCHEDULER_H_*/
