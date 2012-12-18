//-----------------------------------------------------------------------------
// eoGnuplot1DMonitor.h
// (c) Marc Schoenauer, Maarten Keijzer and GeNeura Team, 2000
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
#ifndef EO_eoGnuplot1DMonitor_H
#define EO_eoGnuplot1DMonitor_H

#include <fstream>
#include <string>

#include "eoObject.h"
#include "utils/eoFileMonitor.h"
#include "utils/eoGnuplot.h"
#include "utils/pipecom.h"

/** Plot eoStat

@author Marc Schoenauer
@version 0.0 (2000)

This class plots through gnuplot the eoStat given as argument

eoGnuplot1DMonitor plots stats through gnuplot

Assumes that the same file is appened every so and so, and replots it
everytime

@ingroup Monitors
*/
class eoGnuplot1DMonitor : public eoFileMonitor, public eoGnuplot
{
public:

  // this "using" directive generates a compiler internal error in GCC 4.0.0 ...
  // it's been removed, and the only call to vec was replaced by this->vec in eoGnuplot1DMonitor.cpp
  //    using eoMonitor::vec;

    /** Constructor */
    eoGnuplot1DMonitor(std::string _filename, bool _top=false) :
        eoFileMonitor(_filename, " "),
        eoGnuplot(_filename,(_top?"":"set key bottom"))
        {}

    /** Destructor */
    virtual ~eoGnuplot1DMonitor(){}

    virtual eoMonitor& operator()();

    virtual void FirstPlot();

    /** Class name */
    virtual std::string className() const
        { return "eoGnuplot1DMonitor"; }
};


#endif // EO_eoGnuplot1DMonitor_H



// Local Variables:
// c-file-style: "Stroustrup"
// comment-column: 35
// fill-column: 80
// mode: C++
// End:
