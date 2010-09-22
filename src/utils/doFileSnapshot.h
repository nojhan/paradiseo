//-----------------------------------------------------------------------------
// doFileSnapshot.h
// (c) Marc Schoenauer, Maarten Keijzer and GeNeura Team, 2001
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
             Johann Dreo <johann.dreo@thalesgroup.com>
             Caner Candan <caner.candan@thalesgroup.com>
 */
//-----------------------------------------------------------------------------

#ifndef _doFileSnapshot_h
#define _doFileSnapshot_h

#include <string>
#include <fstream>
#include <stdexcept>

#include "utils/eoMonitor.h"

class doFileSnapshot : public eoMonitor
{
public:

    doFileSnapshot(std::string dirname,
		   unsigned int frequency = 1,
		   std::string filename = "gen",
		   std::string delim = " ",
		   unsigned int counter = 0,
		   bool rmFiles = true,
		   bool saveFilenames = true);

    virtual ~doFileSnapshot();

    virtual bool hasChanged() {return _boolChanged;}
    virtual std::string getDirName() { return _dirname; }
    virtual unsigned int getCounter() { return _counter; }
    virtual const std::string baseFileName() { return _filename;}
    std::string getFileName() {return _currentFileName;}

    void setCurrentFileName();

    virtual eoMonitor& operator()(void);

    virtual eoMonitor& operator()(std::ostream& os);

private :
    std::string _dirname;
    unsigned int _frequency;
    std::string _filename;
    std::string _delim;
    std::string _currentFileName;
    unsigned int _counter;
    bool _saveFilenames;
    std::ofstream* _descOfFiles;
    bool _boolChanged;
};

#endif // !_doFileSnapshot
