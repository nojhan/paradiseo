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

 Here's an example explaning how to use eoLogger:
\code
    #include <eo>

    int	main(int ac, char** av)
    {
    // We are declaring the usual eoParser class
    eoParser parser(ac, av);

    // This call is important to allow -v parameter to change user level.
    make_verbose(parser);

    // At this time we are switching to warning message and messages
    // which are going to follow it are going to be warnings message too.
    // These messages can be displayed only if the user level (sets with
    // eo::setlevel function) is set to eo::warnings.
    eo::log << eo::warnings;

    // With the following eo::file function we are defining that
    // all future logs are going to this new file resource which is
    // test.txt
    eo::log << eo::file("test.txt") << "In FILE" << std::endl;

    // Now we are changing again the resources destination to cout which
    // is the standard output.
    eo::log << std::cout << "In COUT" << std::endl;

    // Here are 2 differents examples of how to set the errors user level
    // in using either a string or an identifier.
    eo::log << eo::setlevel("errors");
    eo::log << eo::setlevel(eo::errors);

    // Now we are writting a message, that will be displayed only if we are above the "quiet" level
    eo::log << eo::quiet << "1) Must be in quiet mode to see that" << std::endl;

    // And so on...
    eo::log << eo::setlevel(eo::warnings) << eo::warnings << "2) Must be in warnings mode to see that" << std::endl;

    eo::log << eo::setlevel(eo::logging);

    eo::log << eo::errors;
    eo::log << "3) Must be in errors mode to see that";
    eo::log << std::endl;

    eo::log << eo::debug << 4 << ')'
    << " Must be in debug mode to see that\n";

    return 0;
    }
\endcode

@{
*/

#ifndef eoLogger_h
#define eoLogger_h

#include <map>
#include <vector>
#include <string>
#include <iosfwd>

#include "eoObject.h"
#include "eoParser.h"

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
     */
    struct file
    {
        file(const std::string f);
        const std::string _f;
    };

    /**
     * setlevel
     * this structure combined with the friend operator<< below is an easy way to set a verbose level.
     */
    struct setlevel
    {
        setlevel(const std::string v);
        setlevel(const Levels lvl);
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
    eoLogger(eo::file file);

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

private:
    /**
     * outbuf
     * this class inherits from std::streambuf which is used by eoLogger to write the buffer in an output stream
     */
    class outbuf : public std::streambuf
    {
    public:
        outbuf(const int& fd, const eo::Levels& contexlvl, const eo::Levels& selectedlvl);
    protected:
        virtual int overflow(int_type c);
    private:
        const int& _fd;
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
     * operator<< used there to set a filename through the class file.
     */
    //! in order to use stream style to define a file to dump instead the standart output
    friend eoLogger& operator<<(eoLogger&, eo::file);

    /**
     * operator<< used there to set a verbose level through the class setlevel.
     */
    //! in order to use stream style to define manually the verbose level instead using options
    friend eoLogger& operator<<(eoLogger&, eo::setlevel);

    /**
     * operator<< used there to be able to use std::cout to say that we wish to redirect back the buffer to a standard output.
     */
    //! in order to use stream style to go back to a standart output defined by STL
    //! and to get retro-compatibility
    friend eoLogger& operator<<(eoLogger&, std::ostream&);

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
     * _fd in storing the file descriptor at this place we can disable easily the buffer in
     * changing the value at -1. It is used by operator <<.
     */
    int _fd;

    /**
     * _obuf std::ostream mandates to use a buffer. _obuf is a outbuf inheriting of std::streambuf.
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
