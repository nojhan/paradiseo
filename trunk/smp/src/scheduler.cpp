/*
<scheduler.cpp>
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

template<class EOT>
paradiseo::smp::Scheduler<EOT>::Scheduler(unsigned workersNb) :
    workers(workersNb),
    popPackages(workersNb),
    done(false),
    planning(workersNb),
    idWaitingThread(-1)
{ }

template<class EOT>
paradiseo::smp::Scheduler<EOT>::~Scheduler()
{ }

template<class EOT>
void paradiseo::smp::Scheduler<EOT>::operator()(eoUF<EOT&, void>& func, eoPop<EOT>& pop)
{
    done = false;
    idWaitingThread = -1;
        
    for(unsigned i = 0; i < workers.size(); i++)
    {
        planning[i] = 2;
        workers[i].start(&Scheduler<EOT>::apply, this,  std::ref(func), std::ref(popPackages[i]), i);
    }

    unsigned counter = 0;
    unsigned j = 0;
        
    while(counter < pop.size())
    {
        std::unique_lock<std::mutex> lock(m);
        cvt.wait(lock, [this]() -> bool {return (int)idWaitingThread != -1;});
        j = 0;
        while (j < planning[idWaitingThread] && counter < pop.size()) 
        {
            popPackages[idWaitingThread].push_back(&pop[counter]);
            counter++;
            j++;
        }
        planning[idWaitingThread] *= 2;
        idWaitingThread = -1;
         cv.notify_one(); 
    }
        
    done = true;
    idWaitingThread = -1;
    cv.notify_all();
        
    for(unsigned i = 0; i < workers.size(); i++)
        workers[i].join();
}

template<class EOT>
void paradiseo::smp::Scheduler<EOT>::apply(eoUF<EOT&, void>& func, std::vector<EOT*>& pop, int id)
{ 
    while(!done || !pop.empty())
    {   
        for(unsigned i = 0; i < pop.size(); i++)
            func(*pop[i]);
        pop.clear();
        std::unique_lock<std::mutex> lock(m);
        idWaitingThread = id;
        // We notify the scheduler we finished the package
        cvt.notify_one();
        // We wait for a new package
        cv.wait(lock, [this]() -> bool {return (int)idWaitingThread == -1 || (bool)done;});
    }
}
