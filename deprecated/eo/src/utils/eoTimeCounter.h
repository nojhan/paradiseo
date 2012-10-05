// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoTimeCounter.h
// (c) Marc Schoenauer, Maarten Keijzer, and GeNeura Team, 2002
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
             Marc.Schoenauer@inria.fr
             mkeijzer@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef _eoTimeCounter_h
#define _eoTimeCounter_h

#include <time.h>
#include <utils/eoStat.h>


/**
    An eoStat that simply gives the user time since first generation
    It has to be tempatized by EOT because it must be an eoStat

    @ingroup Stats
*/
class eoTimeCounter : public eoUpdater, public eoValueParam<double>
{
public:
  eoTimeCounter() : eoValueParam<double>(0.0, "Time") // : firstTime(true)
  {
    start = time(NULL);
  }

  /** simply stores the time spent in process in its value() */
  virtual void operator()()
  {
    // ask for system time
    utime = clock();

    //     if (firstTime)	/* first generation */
    //       {
    //  firstTime=false;
    //  firstUtime = tmsStruct.tms_utime;
    //       }
    // store elapsed user time
    //     value(tmsStruct.tms_utime - firstUtime);
    // value()=double(utime)/CLOCKS_PER_SEC;
    double seconds_elapsed = time(NULL) - start;

    value() = (seconds_elapsed > 2140) ? seconds_elapsed : double(utime)/CLOCKS_PER_SEC;
  }

private:
//   bool firstTime;
//   clock_t firstUtime;
  clock_t utime;
  time_t start;
};

#endif
