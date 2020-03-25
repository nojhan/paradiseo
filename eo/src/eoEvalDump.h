
/*
   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation;
   version 2 of the License.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

   © 2012 Thales group

    Authors:
        Johann Dreo <johann.dreo@thalesgroup.com>
*/

#ifndef eoEvalDump_H
#define eoEvalDump_H

#include <fstream>

#include "eoEvalFunc.h"

/**
Dump an evaluated individual to a given file.

Note: test if the file could be open only in debug mode
If the file cannot be open during the calls, everything will fail in an standard exception.

The file name should be given at instanciation, if you asked for a single file, it will
erase the previously written one and write the new individual in it.

If you do not ask for a single file, it will create several files, 
one different each time it found a individual. The filenames have then the form: 
    <filename><count_prefix><count>
"<count>" being a integer, incremented by one each time a new file is written down.

If you asked for a filename="RESULT", by default, the first dump file will be named "RESULT.0".
*/
template<class EOT>
class eoEvalDump : public eoEvalFunc<EOT>
{
public:
        //! A constructor for wrapping your own evaluator in a eoEvalDump.
        eoEvalDump(
                eoEvalFunc<EOT>& func, std::string filename, bool single_file = false,
                unsigned int file_count = 0, std::string count_prefix = "."
            ) :
            _func(func),
            _filename(filename), _single_file(single_file), _file_count(file_count), _sep(count_prefix),
            _of()
        {}

        //! A constructor without an eval func, the eoEvalDump will thus just write to the file, without evaluating
        eoEvalDump(
                std::string filename, bool single_file = false,
                unsigned int file_count = 0, std::string count_prefix = "."
            ) :
            _dummy_eval(), _func(_dummy_eval),
            _filename(filename), _single_file(single_file), _file_count(file_count), _sep(count_prefix),
            _of()
        {}

        virtual void operator()(EOT& sol)
        {
            _func( sol );
            dump(sol);
        }

    unsigned int file_count() { return _file_count; }

protected:

    // FIXME on x86-64, when called inside a ofstream::open, this function call returns a corrupted string !??
    /*
    const char * filename()
    {
        if( _single_file ) {
            return _filename.c_str();

         } else {
            std::ostringstream afilename;
            afilename << _filename << _sep << _file_count;
            return afilename.str().c_str();
         }
    }
    */

    void dump( EOT & sol )
    {
        if( _single_file ) {
             // explicitely erase the file before writing in it
             _of.open( _filename.c_str(), std::ios_base::out | std::ios_base::trunc );

         } else {
            std::ostringstream afilename;
            afilename << _filename << _sep << _file_count;
            _of.open( afilename.str().c_str() /* NOTE : defaults to : , std::ios_base::out | std::ios_base::trunc  */);
         }
#ifndef NDEBUG
        if ( !_of.is_open() ) {
            std::string str = "Error, eoEvalDump could not open: " + _filename;
            throw std::runtime_error( str );
        }
#endif
        // here, in release mode, we assume that the file could be opened
        // thus, we avoid a supplementary test in this costly evaluator
        _of << sol << std::endl;
        _of.close();

        _file_count++;
    }

protected:
    class DummyEval : public eoEvalFunc<EOT>
    {
        void operator()(EOT&) {/*empty*/}
    };
    DummyEval _dummy_eval;
    eoEvalFunc<EOT>& _func;
    std::string _filename;
    bool _single_file;
    unsigned int _file_count;
    std::string _sep;
    std::ofstream _of;
};

#endif // eoEvalDump_H
