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


#ifndef EO_eoGnuplot1DSnapshot_H
#define EO_eoGnuplot1DSnapshot_H

#include <fstream>
#include <sstream>
#include <string>

#include <eoObject.h>
#include "eoRealVectorBounds.h"
#include <utils/pipecom.h>
#include <utils/eoFileSnapshot.h>
#include <utils/eoGnuplot.h>

/** Plot stats through gnuplot

@author Marc Schoenauer 2000
@version 0.0

This class plots through gnuplot the eoStat given as argument

Assumes that the same file is re-written every so and so, and plots it
from scratch everytime it's called

@ingroup Monitors
 */
class eoGnuplot1DSnapshot: public eoFileSnapshot, public eoGnuplot
{
 public:
    // Ctor
  eoGnuplot1DSnapshot(std::string _dirname, unsigned _frequency = 1,
                      std::string _filename = "gen", std::string _delim = " ",
                      unsigned _counter = 0, bool _rmFiles = true) :
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

  virtual eoMonitor& operator()();

  /// Class name.
  virtual std::string className() const { return "eoGnuplot1DSnapshot"; }

  virtual void handleBounds(eoRealVectorBounds & _bounds)
  {
      std::ostringstream os;
      os << "set autoscale\nset yrange [" ;
      if (_bounds.isMinBounded(0))
          os << _bounds.minimum(0);
      os << ":" ;
      if (_bounds.isMaxBounded(0))
          os << _bounds.maximum(0);
      os << "]\n";
      gnuplotCommand(os.str());
  }

  void setPointSize(unsigned  _pointSize)
  { pointSize = _pointSize; }

protected:

    unsigned pointSize;
};


#endif
