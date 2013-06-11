// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

/*

(c) Thales group, 2010

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

Contact: http://eodev.sourceforge.net

Authors:
Johann Dréo <johann.dreo@thalesgroup.com>
Caner Candan <caner.candan@thalesgroup.com>

*/

/** @defgroup Logging Logging
 * @ingroup Utilities

 Global logger for EO.

 For an example explaning how to use eoLogger, please refer to paradiseo/eo/test/t-eoLogger.cpp

@{
*/

#ifndef eoLogger_h
#define eoLogger_h

#include <map>
#include <vector>
#include <string>
#include <iosfwd>
#include <fstream>

#include "eoObject.h"
#include "eoParser.h"

#define USE_SET
#undef USE_SET

#ifdef USE_SET
#include <set>
#endif


namespace eo
{
    /**
     * Levels contents all the available levels in eoLogger
     *
     * /!\ If you want to add a level dont forget to add it at the implementation of the class eoLogger in the ctor
     */
    enum Levels {quiet = 0,
                 errors,
                 warnings,
                 progress,
                 logging,
                 debug,
                 xdebug};

    /**
     * file
     * this structure combined with the friend operator<< below is an easy way to select a file as output.
     *
    struct file
    {
        explicit file(const std::string f);
        const std::string _f;
    };*/

    /**
     * setlevel
     * this structure combined with the friend operator<< below is an easy way to set a verbose level.
     */
    struct setlevel
    {
        explicit setlevel(const std::string v);
        explicit setlevel(const Levels lvl);
        const std::string _v;
        const Levels _lvl;
    };
}

/**
 * eoLogger
 * Class providing a verbose management through EO
 * Use of a global variable eo::log to easily use the logger like std::cout
 */
class eoLogger : public eoObject,
		 public std::ostream
{
public:
    //! default ctor
    eoLogger();

    //! overidded ctor in order to instanciate a logger with a file define in parameter
    //eoLogger(eo::file file);

    //! dtor
    ~eoLogger();

    //! common function for all eo objects
    virtual std::string className() const;

    //! Print the available levels on the standard output
    //! enablable with the option -l
    void printLevels() const;

    /*! Returns the selected levels, that is the one asked by the user
     *
     * Use this function if you want to be able to compare selected levels to a given one, like:
     * if( eo::log.getLevelSelected() >= eo::progress ) {...}
     */
    inline eo::Levels getLevelSelected() const { return _selectedLevel; }

    /*! Returns the current level of the context
     * the one given when you output message with the logger
     */
    inline eo::Levels getLevelContext() const { return _contextLevel; }

protected:
    //! in order to add a level of verbosity
    void addLevel(std::string name, eo::Levels level);

private:
    //! used by the function make_verbose in order to add options to specify the verbose level
    void _createParameters( eoParser& );

    //! used by the set of ctors to initiate some useful variables
    void _init();

    //! helper function regrouping redirection operations; takes a pointer because can be given NULL
    void doRedirect(std::ostream*);

    //! helper function searching for a stream and removing it, returning true if successful, false otherwise
    bool tryRemoveRedirect(std::ostream*);

private:
    /**
     * outbuf
     * this class inherits from std::streambuf which is used by eoLogger to write the buffer in an output stream
     */
    class outbuf : public std::streambuf
    {
    public:
        outbuf(const eo::Levels& contexlvl, const eo::Levels& selectedlvl);
        //std::ostream * _outStream;
    
    #ifdef USE_SET
        std::set<std::ostream*> _outStreams;
    #else
        std::vector<std::ostream*> _outStreams;
    #endif
    
        std::ofstream * _ownedFileStream;
    protected:
        virtual int overflow(int_type c);
    private:
        const eo::Levels& _contextLevel;
        const eo::Levels& _selectedLevel;
    };

private:
    /**
     * MapLevel is the type used by the map member _levels.
     */
    typedef std::map<std::string, eo::Levels> MapLevel;

public:
    /**
     * operator<< used there to set a verbose mode.
     */
    //! in order to use stream style to define the context verbose level where the following logs will be saved
    friend eoLogger& operator<<(eoLogger&, const eo::Levels);

    /**
     * operator<< used there to set a verbose level through the class setlevel.
     */
    //! in order to use stream style to define manually the verbose level instead using options
    friend eoLogger& operator<<(eoLogger&, eo::setlevel);

    /**
     * DEPRECATED: Use instead the redirect or addRedirect method; has the same effect as addRedirect
     */
    friend eoLogger& operator<<(eoLogger&, std::ostream&);

    /**
     * Redirects the logger to a given output stream.
     * Closing the stream and returning its memory is at the charge of the caller,
     * but should not be done while the log is still redirected to it.
     * This resets all previous redirections set up with redirect and add_redirect.
     */
    void redirect(std::ostream&);

    /**
     * Adds a redirection from the logger to a given output stream.
     * Closing the stream and returning its memory is at the charge of the caller,
     * but should not be done while the log is still redirected to it.
     * This does not reset previous redirections, which remain active.
     */
    void addRedirect(std::ostream&);

    /**
     * Removes a redirection from the logger to the given output stream.
     * This only resets the redirection to the stream passed in argument.
     */
    void removeRedirect(std::ostream&);

    /**
     * Redirects the logger to a file using a filename.
     * Closing the file will be done automatically when the logger is redirected again or destroyed.
     */
    void redirect(const char * filename);
    void redirect(const std::string& filename);

private:
    friend void make_verbose(eoParser&);

    eoValueParam<std::string> _verbose;
    eoValueParam<bool> _printVerboseLevels;
    eoValueParam<std::string> _output;

    /**
     * _selectedLevel is the member storing verbose level setted by the user thanks to operator()
     */
    eo::Levels _selectedLevel;
    eo::Levels _contextLevel;

    /**
     * _obuf std::ostream mandates to use a buffer. _obuf is a outbuf inheriting from std::streambuf.
     */
    outbuf _obuf;

    /**
     * _levels contains all the existing level order by position
     */
    MapLevel _levels;

    /**
     * _levelsOrder is just a list to keep the order of levels
     */
    std::vector<std::string> _sortedLevels;

    /**
     * _standard_io_streams
     */
    std::map< std::ostream*, int > _standard_io_streams;
};
/** @example t-eoLogger.cpp
 */

//! make_verbose gets level of verbose and sets it in eoLogger
void make_verbose(eoParser&);

namespace eo
{
    /**
     * log is an external global variable defined to easily use a same way than std::cout to write a log.
     */
    extern eoLogger log;
}

/** @} */

#endif // !eoLogger_h
