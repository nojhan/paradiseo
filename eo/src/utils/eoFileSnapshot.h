// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoFileSnapshot.h
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
 */
//-----------------------------------------------------------------------------

#ifndef _eoFileSnapshot_h
#define _eoFileSnapshot_h

#include <string>
#include <fstream>
#include <utils/eoParam.h>
#include <utils/eoMonitor.h>
#include <eoObject.h>


/**
    Prints snapshots of fitnesses to a (new) file every N generations 

Assumes that the parameters that are passed to the monitor 
(method add in eoMonitor.h) are eoValueParam<vector<double> > of same size.

A dir is created and one file per snapshot is created there - 
so you can later generate a movie!

TODO: The counter is handled internally, but this should be changed 
so that you can pass e.g. an evalcounter (minor)

I failed to templatize everything so that it can handle eoParam<vector<T> >
for any type T, simply calling their getValue method ...
*/

class eoFileSnapshot : public eoMonitor
{
public :
  typedef vector<double> vDouble;
  typedef eoValueParam<vector<double> > vDoubleParam;

  eoFileSnapshot(std::string _dirname, unsigned _frequency = 1, 
	     std::string _filename = "gen", std::string _delim = " "):
    dirname(_dirname), frequency(_frequency), 
    filename(_filename), delim(_delim), counter(0), boolChanged(true)
  {
    string s = "test -d " + dirname;
    int res = system(s.c_str());
    // test for (unlikely) errors
    if ( (res==-1) || (res==127) )
      throw runtime_error("Problem executing test of dir in eoFileSnapshot");
    // now make sure there is a dir without any genXXX file in it
    if (res)                    // no dir present
      {
	s = string("mkdir ")+dirname; 
      }
    else
      {
	s = string("/bin/rm ")+dirname+ "/" + filename + "*"; 
      }
    system(s.c_str());
    // all done
  }
  
  /** accessor: has something changed (for gnuplot subclass)
   */
  virtual bool hasChanged() {return boolChanged;}

  /** accessor to the current filename: needed by the gnuplot subclass
   */
  string getFileName() {return currentFileName;}

  /** sets the current filename depending on the counter
   */
  void setCurrentFileName()
  {
    char buff[255];
    ostrstream oscount(buff, 254);
    oscount << counter;
    oscount << std::ends; 
    currentFileName = dirname + "/" + filename + oscount.str();
  }

  /** The operator(void): opens the ostream and calls the write method
   */
  eoMonitor& operator()(void)
  {
    if (counter % frequency)
      {
	boolChanged = false;  // subclass with gnuplot will do nothing
	counter++;
	return (*this);
      }
    counter++;
    boolChanged = true;
    setCurrentFileName();
    ofstream os(currentFileName.c_str());
    
    if (!os)
      {
        string str = "eoFileSnapshot: Could not open " + currentFileName;
        throw runtime_error(str);
      }
    
    return operator()(os);
  }

  /** The operator(): write on an ostream
   */
  eoMonitor& operator()(std::ostream& _os)
  {
    const eoValueParam<vector<double> >  * ptParam = 
      static_cast<const eoValueParam<vector<double> >* >(vec[0]);

    const vector<double>  v = ptParam->value();
    if (vec.size() == 1)	   // only one vector: -> add number in front
      {
	for (unsigned k=0; k<v.size(); k++)
	  _os << k << " " << v[k] << "\n" ;
      }
    else			   // need to get all other vectors
      {
	vector<vector<double> > vv(vec.size());
	vv[0]=v;
	cout << "taille des vecteurs " << v.size() << endl;
	for (unsigned i=1; i<vec.size(); i++)
	  {
	    ptParam = static_cast<const eoValueParam<vector<double> >* >(vec[1]);
	    vv[i] = ptParam->value();
	    if (vv[i].size() != v.size())
	      throw runtime_error("Dimension error in eoSnapshotMonitor");
	  }
	for (unsigned k=0; k<v.size(); k++)
	  {
	  for (unsigned i=0; i<vec.size(); i++)
	    _os << vv[i][k] << " " ;
	  _os << "\n";
	  }
      }
    return *this;
   }

  virtual const string getDirName()	   // for eoGnuPlot
  { return dirname;}
  virtual const string baseFileName()	   // the title for eoGnuPlot
  { return filename;}
private :
  std::string dirname;
  unsigned frequency;
  std::string filename;
  std::string delim;
  unsigned int counter;
  std::string currentFileName;
  bool boolChanged;
};

#endif
