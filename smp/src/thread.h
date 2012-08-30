/*
<thread.h>
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

#ifndef THREAD_H_
#define THREAD_H_

#include <iostream>
#include <thread>
#include <vector>

namespace paradiseo
{
namespace smp
{
/** 
A thread class which encapsulate the std::thread behaviour for more flexibility
@see smp::Worker, smp::MWModel
*/
class Thread
{
public:
    /**
     * Default constructor
     */
    Thread() = default;
    
    /**
     * Constructor that will immediatly start the thread
     * @param f represente any callable object such as function or class method
     * @param args... reference object and parameters for f
     */
    template< class Callable, class... Args >
    explicit Thread(Callable&& f, Args&&... args) : t(std::thread(std::forward<Callable>(f), std::forward<Args>(args)...)) {}
    
    Thread(Thread&& other);
    
    Thread(const Thread&) = delete;
    Thread& operator=(const Thread&) = delete;

    virtual ~Thread() = default;
    
    Thread& operator=(Thread&& other);
    
    /**
     * Start the thread according to parameters
     * If the thread is running, it will wait until the end of its task
     * @param f represente any callable object such as function or class method
     * @param args... reference object and parameters for f
     */
    template<class Callable,class... Args>
    void start(Callable&& f,Args&&... args);
    
    /**
     * Get the id of the thread
     * @return id of the thread
     */
    const std::thread::id getId() const;
    
    /**
     * Get the state of the thread
     * @return true if the thread is running, false otherwise
     */
    bool joinable() const;
    
    /**
     * Wait until the end of the task
     */
    void join();
    
    static int hardware_concurrency();

protected:
    std::thread t;
};

template<class Callable,class... Args>
    void paradiseo::smp::Thread::start(Callable&& f,Args&&... args) 
    {
        t = std::thread(std::forward<Callable>(f), std::forward<Args>(args)...);     
    }

}

}

#endif /*THREAD_H_*/
