// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoSeconsElapsedContinue.h
// (c) Maarten Keijzer, 2007
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
 */
//-----------------------------------------------------------------------------

#ifndef _eoTenTimeContinue_h
#define _eoTenTimeContinue_h

#include <time.h>

#include <iostream>
#include <sstream>
#include <cfstream>
#include <eoContinue.h>
/** 
    Timed continuator: continues until a number of seconds is used
*/
template< class EOT>
class eoTenTimeContinue: public eoContinue<EOT>
{

public:

  eoTenTimeContinue(int _maxTime, std::string _fileName) : start(time(0)), maxTime(_maxTime), id(1), fileName(_fileName) {}

  virtual bool operator() ( const eoPop<EOT>& _pop ) {
        time_t diff = time(0) - start;

        
        if (diff > (id * maxTime/10) ){
            time_t begin=time(0);
        	//traitement
            std::ostringstream os;
            os << fileName << "." << id;
        	ofstream outfile(os.str(), ios::app);
    	        
	        for(unsigned int i=0 ; i < _pop.size(); i++){
	        	for(unsigned int j=0 ; j<EOT::ObjectiveVector::nObjectives(); j++){
	        		outfile << finalArchive[i].objectiveVector()[j];
	        		if(j != EOT::ObjectiveVector::nObjectives() -1)
	        			outfile << " ";
	        	}
	        	outfile << endl;
	        }
	              
	        outfile.close();
    	
        	id++;
        	start-=(time(0)-begin);
        }
        if(diff >= maxTime)
        	return false;
        return true;

    }

  
  virtual std::string className(void) const { return "eoTenTimeContinue"; }

  void readFrom (std :: istream & __is) {
    
    __is >> start; 
  }
  
  void printOn (std :: ostream & __os) const {
    
    __os << start << ' ' << std :: endl;    
  }
private:
    time_t start;
    unsigned int id;
    unsigned int maxTime;
    std::string fileName;
};

#endif

