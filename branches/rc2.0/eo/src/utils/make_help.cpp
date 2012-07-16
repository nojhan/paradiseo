//-----------------------------------------------------------------------------
// make_help.h
// (c) Maarten Keijzer, Marc Schoenauer and GeNeura Team, 2001
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
#ifdef _MSC_VER
// to avoid long name warnings
#pragma warning(disable:4786)
#endif

#include <cstdlib>
#include <fstream>
#include <stdexcept>

#include <utils/eoParser.h>

using namespace std;

/** Generation of the status file, and output of the help message if needed
 *
 * MUST be called after ALL parameters have been read in order to list them
 *
 * Warning: this is a plain .cpp file and shoudl NOT be included anywhere,
 * but compiled separately and stored in a library.
 *
 * It is declared in all make_xxx.h files in representation-dependent dirs
 * but it is NOT representation-dependent itself - that's why it's in utils
 */
void make_help(eoParser & _parser)
{
    // name of the "status" file where all actual parameter values will be saved
    string str_status = _parser.ProgramName() + ".status"; // default value
    eoValueParam<string>& statusParam = _parser.createParam(str_status, "status","Status file",'\0', "Persistence" );

    // dump status file BEFORE help, so the user gets a chance to use it:
    // it's probably the case where she/he needs it most!!!
    // Only help parameter will not be in status file - but who cares???
    if (statusParam.value() != "")
      {
        ofstream os(statusParam.value().c_str());
        os << _parser;		// and you can use that file as parameter file
      }
   // do the following AFTER ALL PARAMETERS HAVE BEEN PROCESSED
   // i.e. in case you need parameters somewhere else, postpone these
    if (_parser.userNeedsHelp())
      {
        _parser.printHelp(cout);
        cout << "You can use an edited copy of file " << statusParam.value()
             << " as parameter file" << endl;
        exit(1);
      }
}

/** test a dir.
 *  Creates it if does not exist
 *  If exists, throws an exception or erase everything there,
 *     depending on last parameter
 *
 * Always return true (for code easy writing on the other side :-)
 */
bool testDirRes(std::string _dirName, bool _erase=true)
{
  string s = "test -d " + _dirName;
  int res = system(s.c_str());
  // test for (unlikely) errors
  if ( (res==-1) || (res==127) )
    {
      s = "Problem executing test of dir " + _dirName;
      throw runtime_error(s);
    }
  // now make sure there is a dir without any file in it - or quit
  if (res)                    // no dir present
    {
      s = string("mkdir ")+ _dirName;
      int res = system(s.c_str());
      (void)res;
      return true;
    }
  //  else
  if (_erase)                      // OK to erase
    {
      s = string("/bin/rm ")+ _dirName + "/*";
      int res = system(s.c_str());
      (void)res;
      return true;
    }
  //else
  // WARNING: bug if dir exists and is empty; this says it is not!
  // shoudl use scandir instead - no time now :-(((    MS Aug. 01
  s = "Dir " + _dirName + " is not empty";
  throw runtime_error(s);
  return true;
}


// Local Variables:
// coding: iso-8859-1
// mode: C++
// c-file-offsets: ((c . 0))
// c-file-style: "Stroustrup"
// fill-column: 80
// End:
