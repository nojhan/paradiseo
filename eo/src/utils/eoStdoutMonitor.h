// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoStdoutMonitor.h
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

#ifndef _eoStdoutMonitor_h
#define _eoStdoutMonitor_h

#include <string>

#include <utils/eoMonitor.h>
#include <eoObject.h>

/**
    Prints statistics to stdout
*/
class eoStdoutMonitor : public eoMonitor
{
public :
    eoStdoutMonitor(bool _verbose=true, std::string _delim = "\t") : 
      verbose(_verbose), delim(_delim), firsttime(true) {}
    eoMonitor& operator()(void);

  virtual std::string className(void) const { return "eoStdoutMonitor"; }
private :
  bool verbose;
  std::string delim;
  bool firsttime;
};

#endif
