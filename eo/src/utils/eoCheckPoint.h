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

template <class EOT> class eoStatBase;
template <class EOT> class eoSortedStatBase;
class eoMonitor;
class eoUpdater;

template <class EOT>
class eoCheckPoint : public eoContinue<EOT>
{
public :

    eoCheckPoint(eoContinue<EOT>& _cont) : cont(_cont) {}

    bool operator()(const eoPop<EOT>& _pop);

    void add(eoSortedStatBase<EOT>& _stat) { sorted.push_back(&_stat); }
    void add(eoStatBase<EOT>& _stat) { stats.push_back(&_stat); }
    void add(eoMonitor& _mon)        { monitors.push_back(&_mon); }
    void add(eoUpdater& _upd)        { updaters.push_back(&_upd); }

    std::string className(void) const { return "eoCheckPoint"; }

private :

    eoContinue<EOT>& cont;
    std::vector<eoSortedStatBase<EOT>*>    sorted;
    std::vector<eoStatBase<EOT>*>    stats;
    std::vector<eoMonitor*> monitors;
    std::vector<eoUpdater*> updaters;
};

template <class EOT>
bool eoCheckPoint<EOT>::operator()(const eoPop<EOT>& _pop)
{
    unsigned i;

    if (!sorted.empty())
    {
      vector<const EOT*> sorted_pop;
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

    return cont(_pop);
}

#endif