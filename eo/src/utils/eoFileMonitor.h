// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoFileMonitor.h
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
CVS Info: $Date: 2004-09-21 19:49:48 $ $Header: /home/nojhan/dev/eodev/eodev_cvs/eo/src/utils/eoFileMonitor.h,v 1.13 2004-09-21 19:49:48 maartenkeijzer Exp $ $Author: maartenkeijzer $ 

 */
//-----------------------------------------------------------------------------
/** Modified the default behavior, so that it erases existing files.
Can be modified in the ctor.
MS 25/11/00
*/

#ifndef _eoFileMonitor_h
#define _eoFileMonitor_h

#include <string>
#include <fstream>

#include <utils/eoMonitor.h>
#include <eoObject.h>

/**
    Prints statistics to file
*/
class eoFileMonitor : public eoMonitor
{
public :
  
  eoFileMonitor(std::string _filename, std::string _delim = " ", bool _keep = false, bool _header=false) 
    : filename(_filename), delim(_delim), keep(_keep), header(_header), firstcall(true) 
  {
    if (! _keep) {
      std::ofstream os(filename.c_str());
      if (!os){
	std::string str = "eoFileMonitor: Could not open " + filename;
	throw std::runtime_error(str);
      }
    }
  }
  
  virtual eoMonitor& operator()(void);

  virtual eoMonitor& operator()(std::ostream& os);

  void printHeader(void);
  virtual void printHeader(std::ostream& os);
  
  virtual std::string getFileName() { return filename;}
  
private :
  std::string filename;
  std::string delim;
  bool keep;			   // should we append or create a new file
  bool header;                     // printing header at begin of file?
  bool firstcall;
  
};

#endif
