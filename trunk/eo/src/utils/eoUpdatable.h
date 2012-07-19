// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoUpdatable.h
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

#ifndef _eoUpdatable_h
#define _eoUpdatable_h

#include <utils/eoUpdater.h>

/**
    eoUpdatable is a generic class for adding updatation to an existing class
    Just says it has an update() method

    @ingroup Utilities
*/
class eoUpdatable
{
public:

    /** @brief Virtual destructor */
    virtual ~eoUpdatable() {};

    virtual void update() = 0;
};



/**
   A base class to actually update an eoUpdatable object

    @ingroup Utilities
*/
class eoDynUpdater : public eoUpdater
{public :
    eoDynUpdater(eoUpdatable & _toUpdate) : toUpdate(_toUpdate) {};

    virtual void operator()()
    {
        toUpdate.update();
    }

private:
    eoUpdatable& toUpdate;
};

/**
   An eoUpdater to update an eoUpdatable object every given time interval

    @ingroup Utilities
*/
class eoTimedDynUpdate : public eoDynUpdater
{
public :
    eoTimedDynUpdate(eoUpdatable & _toUpdate, time_t _interval) :
    eoDynUpdater(_toUpdate),
        interval(_interval), last_time(time(0)), first_time(time(0)) {}

    void operator()(void)
      {
        time_t now = time(0);

        if (now >= last_time + interval)
          {
            last_time = now;
            eoDynUpdater::operator() ();
          }
      }
private :
    const time_t interval;
    time_t last_time;
    const time_t first_time;
};

/**
   An eoUpdater to update an eoUpdatable object every given tic

    @ingroup Utilities
*/
class eoCountedDynUpdate : public eoDynUpdater
{
public :
    eoCountedDynUpdate(eoUpdatable & _toUpdate, unsigned _interval)
        : eoDynUpdater(_toUpdate), interval(_interval), counter(0) {}

    void operator()(void)
      {
        if (++counter % interval == 0)
          {
            eoDynUpdater::operator() ();
          }
      }
private :
    const unsigned interval;
    unsigned counter;
};

#endif
