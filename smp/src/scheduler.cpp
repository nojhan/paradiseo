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

template<class EOT, class Policy>
paradiseo::smp::Scheduler<EOT,Policy>::Scheduler(unsigned workersNb) :
    workers(workersNb),
    popPackages(workersNb),
    done(false),
    planning(workersNb),
    m(workersNb)
{ }

template<class EOT, class Policy>
paradiseo::smp::Scheduler<EOT,Policy>::~Scheduler()
{ }

template<class EOT, class Policy>
void paradiseo::smp::Scheduler<EOT,Policy>::operator()(eoUF<EOT&, void>& func, eoPop<EOT>& pop)
{
    // Call the tag dispatcher
    operator()(func, pop,typename policyTraits<Policy>::type());
}

template<class EOT, class Policy>
void paradiseo::smp::Scheduler<EOT,Policy>::operator()(eoUF<EOT&, void>& func, eoPop<EOT>& pop, const LinearPolicy&)
{
    // Determine number of packages according to the number of workers
    unsigned nbPackages = workers.size();

    // Fill packages
    unsigned nbIndi = pop.size() / nbPackages;
    unsigned remaining = pop.size() % nbPackages;    
    unsigned indice = 0;
        
    for(unsigned i = 0; i < nbPackages; i++)
    {
        popPackages[i].clear();
        for(unsigned j = 0; j < nbIndi; j++)
        {
            popPackages[i].push_back(&pop[i*nbIndi+j]);
            indice = i*nbIndi+j;
        }
    }
    
    if(nbIndi != 0) // Handle the offset if there is less individuals than workers
        indice++;
    for(unsigned i = 0; i < remaining; i++) 
        popPackages[i].push_back(&pop[indice+i]); 
        
    // Starting threads
    for(unsigned i = 0; i < workers.size(); i++)
        if(!popPackages[i].empty())
            workers[i] = std::thread(&Scheduler<EOT,Policy>::applyLinearPolicy, this,  std::ref(func), std::ref(popPackages[i]));
   
    // Wait the end of tasks
    for(unsigned i = 0; i < workers.size(); i++)
        if(!popPackages[i].empty() && workers[i].joinable())
            workers[i].join();
}

template<class EOT, class Policy>
void paradiseo::smp::Scheduler<EOT,Policy>::operator()(eoUF<EOT&, void>& func, eoPop<EOT>& pop, const ProgressivePolicy&)
{
    done = false;
        
    for(unsigned i = 0; i < workers.size(); i++)
    {
        planning[i] = 2;
        workers[i] = std::thread(&Scheduler<EOT,Policy>::applyProgressivePolicy, this,  std::ref(func), std::ref(popPackages[i]), i);
    }

    unsigned counter = 0;
    unsigned j = 0;
        
    while(counter < pop.size())
    {
        for(unsigned i = 0; i < workers.size(); i++)
        {
            if(popPackages[i].empty() && counter < pop.size())
            {
                j = 0;
                std::unique_lock<std::mutex> lock(m[i]);
                while (j < planning[i] && counter < pop.size()) 
                {
                    //std::cout << counter << std::endl;
                    popPackages[i].push_back(&pop[counter]);
                    counter++;
                    j++;
                }
                planning[i] *= 2;
            }
        }
        
        std::this_thread::sleep_for(std::chrono::nanoseconds(10));
    }
      
    done = true;
    
    for(unsigned i = 0; i < workers.size(); i++)
        workers[i].join();
}

template<class EOT, class Policy>
void paradiseo::smp::Scheduler<EOT,Policy>::applyLinearPolicy(eoUF<EOT&, void>& func, std::vector<EOT*>& pop)
{ 
    for(unsigned i = 0; i < pop.size(); i++)
        func(*pop[i]);
}

template<class EOT, class Policy>
void paradiseo::smp::Scheduler<EOT,Policy>::applyProgressivePolicy(eoUF<EOT&, void>& func, std::vector<EOT*>& pop, int id)
{ 
    while(!done || !pop.empty())
    {   
        std::unique_lock<std::mutex> lock(m[id]);
        for(unsigned i = 0; i < pop.size(); i++) {
            //std::cout << "."  << id << "." << std::endl;
            func(*pop[i]);
        }
        pop.clear();
        lock.unlock();
    }
}
