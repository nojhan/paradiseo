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


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <sstream>

#include "utils/eoGnuplot1DMonitor.h"
#include "utils/eoParam.h"


eoMonitor& eoGnuplot1DMonitor::operator() (void)
{
    // update file using the eoFileMonitor
    eoFileMonitor::operator()();
#ifdef HAVE_GNUPLOT
    // sends plot order to gnuplot
    // assumes successive plots will have same nb of columns!!!
    if (firstTime)
    {
        FirstPlot();
        firstTime = false;
    }
    else
    {
        if( gpCom ) {
            PipeComSend( gpCom, "replot\n" );
        }
    }
#endif
    return *this;
}



void eoGnuplot1DMonitor::FirstPlot()
{
    if (this->vec.size() < 2)
    {
        throw std::runtime_error("Must have some stats to plot!\n");
    }
#ifdef HAVE_GNUPLOT
    std::ostringstream os;
    os << "plot";
    for (unsigned i=1; i<this->vec.size(); i++) {
        os << " '" << getFileName().c_str() <<
            "' using 1:" << i+1 << " title '" << (this->vec[i])->longName() << "' with lines" ;
        if (i<this->vec.size()-1)
            os << ", ";
    }
    os << '\n';
    PipeComSend( gpCom, os.str().c_str());
#endif
}



// Local Variables:
// c-file-style: "Stroustrup"
// fill-column: 80
// End:
