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
             Marc.Schoenauer@polytechnique.fr
             mkeijzer@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef _eoGnuplot1DSnapshot_H
#define _eoGnuplot1DSnapshot_H

#include <string>

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
#include <utils/pipecom.h>



/** eoGnuplot1DMonitor plots stats through gnuplot
 *  assumes that the same file is appened every so and so, 
 *  and replots it everytime
 */
class eoGnuplot1DSnapshot: public eoFileSnapshot, public eoGnuplot
{
 public:
    // Ctor
  eoGnuplot1DSnapshot(std::string _dirname, unsigned _frequency = 1, 
	     std::string _filename = "gen", std::string _delim = " ") : 
      eoFileSnapshot(_dirname, _frequency, _filename, _delim), 
      eoGnuplot(_filename,"set data style points")
  {}

    // Ctor
  eoGnuplot1DSnapshot(eoFileSnapshot & _fSnapshot) : 
      eoFileSnapshot(_fSnapshot), 
      eoGnuplot(_fSnapshot.baseFileName(),"set data style points")
  {}
  
  // Dtor
  virtual ~eoGnuplot1DSnapshot(){}

  virtual eoMonitor&  operator() (void) ;

  /// Class name.
  virtual string className() const { return "eoGnuplot1DSnapshot"; }

private: 
};

// the following should be placed in a separate eoGnuplot1DMonitor.cpp 

////////////////////////////////////////////////////////////
inline eoMonitor&   eoGnuplot1DSnapshot::operator() (void)
  /////////////////////////////////////////////////////////
{
  // update file using the eoFileMonitor
  eoFileSnapshot::operator()();

  // sends plot order to gnuplot
  // assumes successive plots will have same nb of columns!!!


  char buff[1024];
  ostrstream os(buff, 1024);
  os << "plot";

    os << " '" << getFileName().c_str() <<
      "' notitle with points ps 5" ;
  os << "\n";
  os << '\0';
  PipeComSend( gpCom, buff );

  return (*this);
}
    
#endif _eoGnuplot1DSnapshot_H
