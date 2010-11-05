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
	 Johann Dréo <johann.dreo@thalesgroup.com>
	 Caner Candan <caner.candan@thalesgroup.com>
*/

#ifndef EO_PARSER_LOGGER_H
#define EO_PARSER_LOGGER_H

#include "eoParser.h"
#include "eoLogger.h"

/**
 * A parser that use the advanced logging system (@see eoLogger)
 *
 * @ingroup Parameters
 * @ingroup Logging
 */
class	eoParserLogger : public eoParser
{
public:
    eoParserLogger(unsigned _argc, char** _argv,
		   std::string _programDescription = "",
		   std::string _lFileParamName = "param-file",
		   char _shortHand = 'p');
    ~eoParserLogger();

private:
    friend void	make_verbose(eoParserLogger&);
    eoValueParam<std::string>	_verbose;
    eoValueParam<bool>		_printVerboseLevels;
    eoValueParam<std::string>	_output;
};

void	make_verbose(eoParserLogger&);

#endif // !EO_PARSER_LOGGER_H

// Local Variables:
// coding: iso-8859-1
// mode: C++
// c-file-offsets: ((c . 0))
// c-file-style: "Stroustrup"
// fill-column: 80
// End:
