// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoUpdater.h
// (c) Maarten Keijzer, Marc Schoenauer and GeNeura Team, 2000
/* 
    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Contact: todos@geneura.ugr.es, http://geneura.ugr.es
             Marc.Schoenauer@polytechnique.fr
             mkeijzer@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef _eoUpdater_h
#define _eoUpdater_h

#include <eoFunctor.h>
#include <utils/eoState.h>

/**
    eoUpdater is a generic procudere for updating whatever you want.
    Yet again an empty name
*/
class eoUpdater : public eoProcedure<void>
{};

/**
*/
template <class T>
class eoIncrementor : public eoUpdater
{public :   
    eoIncrementor(T& _counter, T _stepsize = 1) : counter(_counter), stepsize(_stepsize) {}

    virtual void operator()()
    {
        counter += stepsize;
    }

private:
    T& counter;
    T stepsize;
};

#include <time.h>

/**
*/
class eoTimedStateSaver : public eoUpdater
{
public :
    eoTimedStateSaver(time_t _interval, const eoState& _state, std::string _prefix = "state", std::string _extension = "sav") : state(_state), 
        interval(_interval), last_time(time(0)), first_time(time(0)),
    prefix(_prefix), extension(_extension) {}

    void operator()(void);

private :
    const eoState& state;

    const time_t interval;
    time_t last_time;
    const time_t first_time;
    const std::string prefix;
    const std::string extension;
};

/**
*/
class eoCountedStateSaver : public eoUpdater
{
public :
    eoCountedStateSaver(unsigned _interval, const eoState& _state, std::string _prefix = "state", std::string _extension = "sav") 
        : state(_state), interval(_interval), counter(0),
    prefix(_prefix), extension(_extension) {}

    void operator()(void);

private :
    const eoState& state;
    const unsigned interval;
    unsigned counter;
    
    const std::string prefix;
    const std::string extension;
};

#endif