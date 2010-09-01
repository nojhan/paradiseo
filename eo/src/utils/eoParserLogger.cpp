/*

(c) Thales group, 2010

    This library is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License as published by the Free
    Software Foundation; version 2 of the license.

    This library is distributed in the hope that it will be useful, but WITHOUT ANY
    WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
    PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License along
    with this library; if not, write to the Free Software Foundation, Inc., 59
    Temple Place, Suite 330, Boston, MA 02111-1307 USA

Contact: http://eodev.sourceforge.net

Authors:
Johann Dreo <johann.dreo@thalesgroup.com>
Caner Candan <caner.candan@thalesgroup.com>

*/

#include "eoParserLogger.h"

eoParserLogger::eoParserLogger(unsigned _argc, char** _argv,
			       std::string _programDescription /*= ""*/
			       ,
			       std::string _lFileParamName /*= "param-file"*/,
			       char _shortHand /*= 'p'*/)
    : eoParser(_argc, _argv, _programDescription, _lFileParamName, _shortHand),
    _verbose("quiet", "verbose", "Set the verbose level", 'v'),
    _printVerboseLevels(false, "print-verbose-levels", "Print verbose levels", 'l'),
    _output("", "output", "Redirect a standard output to a file", 'o')
{
    //------------------------------------------------------------------
    // we are saying to eoParser to create the parameters created above.
    //------------------------------------------------------------------

    processParam(_verbose);
    processParam(_printVerboseLevels);
    processParam(_output);

    //------------------------------------------------------------------


    //------------------------------------------------------------------
    // we're gonna redirect the log to the given filename if -o is used.
    //------------------------------------------------------------------

    if ( ! _output.value().empty() )
	{
	    eo::log << eo::file( _output.value() );
	}

    //------------------------------------------------------------------


    //------------------------------------------------------------------
    // // we're gonna print the list of levels if -l parameter is used.
    //------------------------------------------------------------------

    if ( _printVerboseLevels.value() )
	{
	    eo::log.printLevels();
	}

    //------------------------------------------------------------------

}

eoParserLogger::~eoParserLogger()
{}

//! make_verbose gets level of verbose and sets it in eoLogger
void	make_verbose(eoParserLogger& _parser)
{
    eo::log << eo::setlevel(_parser._verbose.value());
}
