// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoCheckPoint.h
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

#ifndef _eoCheckPoint_h
#define _eoCheckPoint_h

#include <eoContinue.h>
#include <utils/eoUpdater.h>
#include <utils/eoMonitor.h>
#include <utils/eoStat.h>

/** @defgroup Checkpoints Checkpointing
 *
 * Checkpoints are supposed to be called perodically (for instance at each generation) and
 * will call every functors you put in them.
 *
 * Use them with eoStats, eoUpdater and eoMonitor to get statistics at each generation.
 *
 * @see eoStat
 * @see eoMonitor
 * @see eoUpdater
 *
 * Example of a test program using checkpointing:
 * @include t-eoCheckpointing.cpp
 *
 * @ingroup Utilities
 *
 * @{
 */

/** eoCheckPoint is a container class.
    It contains std::vectors of (pointers to)
             eoContinue    (modif. MS July 16. 2002)
             eoStats, eoUpdater and eoMonitor
    it is an eoContinue, so its operator() will be called every generation -
             and will return the contained-combined-eoContinue result
    but before that it will call in turn every single
             {statistics, updaters, monitors} that it has been given,
    and after that, if stopping, all lastCall methods of the above.
*/
template <class EOT>
class eoCheckPoint : public eoContinue<EOT>
{
public :

    eoCheckPoint(eoContinue<EOT>& _cont)
  {
    continuators.push_back(&_cont);
  }

    bool operator()(const eoPop<EOT>& _pop);

    void add(eoContinue<EOT>& _cont) { continuators.push_back(&_cont); }
    void add(eoSortedStatBase<EOT>& _stat) { sorted.push_back(&_stat); }
    void add(eoStatBase<EOT>& _stat) { stats.push_back(&_stat); }
    void add(eoMonitor& _mon)        { monitors.push_back(&_mon); }
    void add(eoUpdater& _upd)        { updaters.push_back(&_upd); }

    virtual std::string className(void) const { return "eoCheckPoint"; }
    std::string allClassNames() const ;

private :

  std::vector<eoContinue<EOT>*>    continuators;
    std::vector<eoSortedStatBase<EOT>*>    sorted;
    std::vector<eoStatBase<EOT>*>    stats;
    std::vector<eoMonitor*> monitors;
    std::vector<eoUpdater*> updaters;
};

template <class EOT>
bool eoCheckPoint<EOT>::operator()(const eoPop<EOT>& _pop)
{
    unsigned i;

    std::vector<const EOT*> sorted_pop;
    if (!sorted.empty())
    {
      _pop.sort(sorted_pop);

      for (i = 0; i < sorted.size(); ++i)
      {
        (*sorted[i])(sorted_pop);
      }
    }

    for (i = 0; i < stats.size(); ++i)
        (*stats[i])(_pop);

    for (i = 0; i < updaters.size(); ++i)
        (*updaters[i])();

    for (i = 0; i < monitors.size(); ++i)
        (*monitors[i])();

    bool bContinue = true;
    for (i = 0; i < continuators.size(); ++i)
      if ( !(*continuators[i])(_pop) )
        bContinue = false;

    if (! bContinue)       // we're going to stop: lastCall, gentlemen
      {
        if (!sorted.empty())
          {
            for (i = 0; i < sorted.size(); ++i)
              {
                sorted[i]->lastCall(sorted_pop);
              }
          }
        for (i = 0; i < stats.size(); ++i)
          stats[i]->lastCall(_pop);

        for (i = 0; i < updaters.size(); ++i)
          updaters[i]->lastCall();

        for (i = 0; i < monitors.size(); ++i)
          monitors[i]->lastCall();
      }
    return bContinue;
}

/** returns a string with all className()
 *  of data separated with "\n" (for debugging)
 */
template <class EOT>
std::string eoCheckPoint<EOT>::allClassNames() const
{
    unsigned i;
    std::string s = "\n" + className() + "\n";

    s += "Sorted Stats\n";
    for (i = 0; i < sorted.size(); ++i)
        s += sorted[i]->className() + "\n";
    s += "\n";

    s += "Stats\n";
    for (i = 0; i < stats.size(); ++i)
        s += stats[i]->className() + "\n";
    s += "\n";

    s += "Updaters\n";
    for (i = 0; i < updaters.size(); ++i)
        s += updaters[i]->className() + "\n";
    s += "\n";

    s += "Monitors\n";
    for (i = 0; i < monitors.size(); ++i)
        s += monitors[i]->className() + "\n";
    s += "\n";

    s += "Continuators\n";
    for (i = 0; i < continuators.size(); ++i)
        s += continuators[i]->className() + "\n";
    s += "\n";

    return s;
}

/** @} */
#endif
