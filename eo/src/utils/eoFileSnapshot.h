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

#include <cstdlib>
#include <fstream>
#include <string>

#include <utils/eoParam.h>
#include <utils/eoMonitor.h>
#include <eoObject.h>


/**
    Prints snapshots of fitnesses to a (new) file every N generations

Assumes that the parameters that are passed to the monitor
(method add in eoMonitor.h) are eoValueParam<std::vector<double> > of same size.

A dir is created and one file per snapshot is created there -
so you can later generate a movie!

@todo The counter is handled internally, but this should be changed
so that you can pass e.g. an evalcounter (minor)

I failed to templatize everything so that it can handle eoParam<std::vector<T> >
for any type T, simply calling their getValue method ...

@ingroup Monitors
*/
class eoFileSnapshot : public eoMonitor
{
public :
  typedef std::vector<double> vDouble;
  typedef eoValueParam<std::vector<double> > vDoubleParam;

  eoFileSnapshot(std::string _dirname, unsigned _frequency = 1, std::string _filename = "gen",
              std::string _delim = " ", unsigned _counter = 0, bool _rmFiles = true):
    dirname(_dirname), frequency(_frequency),
    filename(_filename), delim(_delim), counter(_counter), boolChanged(true)
  {
    std::string s = "test -d " + dirname;

    int res = system(s.c_str());
    // test for (unlikely) errors
    if ( (res==-1) || (res==127) )
      throw std::runtime_error("Problem executing test of dir in eoFileSnapshot");
    // now make sure there is a dir without any genXXX file in it
    if (res)                    // no dir present
      {
        s = std::string("mkdir ")+dirname;
      }
    else if (!res && _rmFiles)
      {
        s = std::string("/bin/rm ")+dirname+ "/" + filename + "*";
      }
    else
      s = " ";

    res=system(s.c_str());
    // all done
  }

  /** accessor: has something changed (for gnuplot subclass)
   */
  virtual bool hasChanged() {return boolChanged;}

  /** accessor to the counter: needed by the gnuplot subclass
   */
  unsigned getCounter() {return counter;}

  /** accessor to the current filename: needed by the gnuplot subclass
   */
  std::string getFileName() {return currentFileName;}

  /** sets the current filename depending on the counter
   */
  void setCurrentFileName()
  {
      std::ostringstream oscount;
      oscount << counter;
      currentFileName = dirname + "/" + filename + oscount.str();
  }

  /** The operator(void): opens the std::ostream and calls the write method
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
    std::ofstream os(currentFileName.c_str());

    if (!os)
      {
        std::string str = "eoFileSnapshot: Could not open " + currentFileName;
        throw std::runtime_error(str);
      }

    return operator()(os);
  }

  /** The operator(): write on an std::ostream
   */
  eoMonitor& operator()(std::ostream& _os)
  {
    const eoValueParam<std::vector<double> >  * ptParam =
      static_cast<const eoValueParam<std::vector<double> >* >(vec[0]);

    const std::vector<double>  v = ptParam->value();
    if (vec.size() == 1)           // only one std::vector: -> add number in front
      {
        for (unsigned k=0; k<v.size(); k++)
          _os << k << " " << v[k] << "\n" ;
      }
    else                           // need to get all other std::vectors
      {
        std::vector<std::vector<double> > vv(vec.size());
        vv[0]=v;
        for (unsigned i=1; i<vec.size(); i++)
          {
            ptParam = static_cast<const eoValueParam<std::vector<double> >* >(vec[1]);
            vv[i] = ptParam->value();
            if (vv[i].size() != v.size())
              throw std::runtime_error("Dimension error in eoSnapshotMonitor");
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

  virtual const std::string getDirName()           // for eoGnuPlot
  { return dirname;}
  virtual const std::string baseFileName()         // the title for eoGnuPlot
  { return filename;}

  /// add checks whether it is a std::vector of doubles
  void add(const eoParam& _param)
  {
    if (!dynamic_cast<const eoValueParam<std::vector<double> >*>(&_param))
    {
      throw std::logic_error(std::string("eoFileSnapshot: I can only monitor std::vectors of doubles, sorry. The offending parameter name = ") + _param.longName());
    }
    eoMonitor::add(_param);
  }

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


// Local Variables:
// coding: iso-8859-1
// mode: C++
// c-file-offsets: ((c . 0))
// c-file-style: "Stroustrup"
// fill-column: 80
// End:
