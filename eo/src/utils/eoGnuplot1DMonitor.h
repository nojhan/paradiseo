// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

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
#ifndef NO_GNUPLOT

#ifndef _eoGnuplot1DMonitor_H
#define _eoGnuplot1DMonitor_H

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string>

#include <utils/eoMonitor.h>
#include <utils/eoGnuplot.h>
#include <eoObject.h>

/**
@author Marc Schoenauer 2000
@version 0.0

This class plots through gnuplot the eoStat given as argument

*/
//-----------------------------------------------------------------------------

#include <fstream>
#include <utils/pipecom.h>



/** eoGnuplot1DMonitor plots stats through gnuplot
 *  assumes that the same file is appened every so and so,
 *  and replots it everytime
 */
class eoGnuplot1DMonitor: public eoFileMonitor, public eoGnuplot
{
 public:
    // Ctor
  eoGnuplot1DMonitor(std::string _filename, bool _top=false) :
      eoFileMonitor(_filename, " "),
      eoGnuplot(_filename,(_top?"":"set key bottom"))
  {}

  // Dtor
  virtual ~eoGnuplot1DMonitor(){}

  virtual eoMonitor&  operator() (void) ;
  virtual void  FirstPlot();

  /// Class name.
  virtual std::string className() const { return "eoGnuplot1DMonitor"; }

private:
};

// the following should be placed in a separate eoGnuplot1DMonitor.cpp
// then the inline specifier should dissappear
////////////////////////////////////////////////////////////
inline eoMonitor&   eoGnuplot1DMonitor::operator() (void)
  /////////////////////////////////////////////////////////
{
  // update file using the eoFileMonitor
  eoFileMonitor::operator()();

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
  return *this;
}

////////////////////////////////////////////////////////////
inline void  eoGnuplot1DMonitor::FirstPlot()
  ////////////////////////////////////////////////////////
{
  if (vec.size() < 2)
    {
      throw std::runtime_error("Must have some stats to plot!\n");
    }
#ifdef HAVE_SSTREAM
  std::ostringstream os;
#else
  char buff[1024];
  std::ostrstream os(buff, 1024);
#endif

  os << "plot";
  for (unsigned i=1; i<vec.size(); i++) {
    os << " '" << getFileName().c_str() <<
      "' using 1:" << i+1 << " title '" << vec[i]->longName() << "' with lines" ;
    if (i<vec.size()-1)
      os << ", ";
  }
  os << '\n';
#ifdef HAVE_SSTREAM
  PipeComSend( gpCom, os.str().c_str());
#else
  os << std::ends;
  PipeComSend( gpCom, buff );
#endif
}

#endif
#endif
