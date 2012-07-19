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

    (c) Thales group 2011

    Author: johann.dreo@thalesgroup.com
 */

#ifndef _eoGenCounter_h
#define _eoGenCounter_h

#include <string>
#include <utils/eoStat.h>

/**
    An eoStat that simply gives the current generation index

    @ingroup Stats
*/
class eoGenCounter : public eoUpdater, public eoValueParam<unsigned int>
{
public:
  eoGenCounter( unsigned int start = 0, std::string label = "Gen" ) : eoValueParam<unsigned int>(start, label), _nb(start) {}

  virtual void operator()()
  {
    value() = _nb++;
  }

private:
  unsigned int _nb;
};

#endif
