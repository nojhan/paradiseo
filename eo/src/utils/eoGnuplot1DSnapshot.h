// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoGnuplot1DSnapshot.h
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
             Marc.Schoenauer@inria.fr
             mkeijzer@dhi.dk
 */
//-----------------------------------------------------------------------------
#ifndef NO_GNUPLOT

#ifndef _eoGnuplot1DSnapshot_H
#define _eoGnuplot1DSnapshot_H

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string>
#ifdef HAVE_SSTREAM
#include <sstream>
#endif

#include <utils/eoFileSnapshot.h>
#include <utils/eoGnuplot.h>
#include <eoObject.h>

/**
@author Marc Schoenauer 2000
@version 0.0

This class plots through gnuplot the eoStat given as argument

*/
//-----------------------------------------------------------------------------

#include <fstream>
#include "eoRealVectorBounds.h"
#include <utils/pipecom.h>



/** eoGnuplot1DSnapshot plots stats through gnuplot
 *  assumes that the same file is re-written every so and so, 
 *  and plots it from scratch everytime it's called
 */
class eoGnuplot1DSnapshot: public eoFileSnapshot, public eoGnuplot
{
 public:
    // Ctor
  eoGnuplot1DSnapshot(std::string _dirname, unsigned _frequency = 1,
	     std::string _filename = "gen", std::string _delim = " ", unsigned _counter = 0, bool _rmFiles = true) :
      eoFileSnapshot(_dirname, _frequency, _filename, _delim, _counter, _rmFiles),
      eoGnuplot(_filename,"set data style points"),
      pointSize(5)
  {}

    // Ctor
  eoGnuplot1DSnapshot(std::string _dirname,  eoRealVectorBounds & _bounds,
		       unsigned _frequency = 1, std::string _filename = "gen",
		       std::string _delim = " ", unsigned _counter = 0, bool _rmFiles = true ) :
      eoFileSnapshot(_dirname, _frequency, _filename, _delim, _counter, _rmFiles),
      eoGnuplot(_filename,"set data style points"),
      pointSize(5)
  {
    handleBounds(_bounds);
  }
    // Ctor
  eoGnuplot1DSnapshot(eoFileSnapshot & _fSnapshot) :
      eoFileSnapshot(_fSnapshot),
      eoGnuplot(_fSnapshot.baseFileName(),"set data style points"),
      pointSize(5)
  {}

    // Ctor with range
  eoGnuplot1DSnapshot(eoFileSnapshot & _fSnapshot, eoRealVectorBounds & _bounds) :
      eoFileSnapshot(_fSnapshot),
      eoGnuplot(_fSnapshot.baseFileName(),"set data style points"),
      pointSize(5)
  {
    handleBounds(_bounds);
  }

  // Dtor
  virtual ~eoGnuplot1DSnapshot(){}

  virtual eoMonitor&  operator() (void) ;

  /// Class name.
  virtual std::string className() const { return "eoGnuplot1DSnapshot"; }

  virtual void handleBounds(eoRealVectorBounds & _bounds)
  {
#ifdef HAVE_SSTREAM
      std::ostringstream os;
#else
    // use strstream and not std::stringstream until strstream is in all distributions
    char buf[1024];
    std::ostrstream os(buf, 1023);
#endif
    //    std::ostrstream os;       
    os << "set autoscale\nset yrange [" ;
    if (_bounds.isMinBounded(0))
      os << _bounds.minimum(0);
    os << ":" ;
    if (_bounds.isMaxBounded(0))
       os << _bounds.maximum(0);
    os << "]\n";
    gnuplotCommand(os.str());
  }

  unsigned pointSize;
private:

};

// the following should be placed in a separate eoGnuplot1DMonitor.cpp

////////////////////////////////////////////////////////////
inline eoMonitor&   eoGnuplot1DSnapshot::operator() (void)
  /////////////////////////////////////////////////////////
{
  // update file using the eoFileMonitor method
  eoFileSnapshot::operator()();

  // sends plot order to gnuplot
#ifdef HAVE_SSTREAM
  //std::string buff; // need local memory
  std::ostringstream os;
#else
  char buff[1024];
  std::ostrstream os(buff, 1024);
#endif
  
  os << "set title 'Gen. " << getCounter() << "'; plot '"
    // mk: had to use getFilename().c_str(), because it seems the string(stream) lib is screwed in gcc3.2
      << getFileName().c_str() << "' notitle with points ps " << pointSize;
  os << std::endl;
  
#ifdef HAVE_SSTREAM
  PipeComSend( gpCom, os.str().c_str());
#else
  os << std::ends;
  PipeComSend( gpCom, buff );
#endif
  return (*this);
}

#endif
#endif
