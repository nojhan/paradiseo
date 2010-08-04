//-----------------------------------------------------------------------------
// t-eoLogger.cpp
//-----------------------------------------------------------------------------

#include <iostream>
#include <utils/eoLogger.h>
#include <utils/eoParserLogger.h>

//-----------------------------------------------------------------------------

int	main(int ac, char** av)
{
    eoParserLogger	parser(ac, av);

    make_verbose(parser);

    eo::log << eo::setlevel(eo::debug);

    eo::log << eo::warnings;

    eo::log << eo::file("test.txt") << "In FILE" << std::endl;
    eo::log << std::cout << "In COUT" << std::endl;

    eo::log << eo::setlevel("errors");
    eo::log << eo::setlevel(eo::errors);

    eo::log << eo::quiet << "1) Must be in quiet mode" << std::endl;

    eo::log << eo::setlevel(eo::warnings) << eo::warnings << "2) Must be in warnings mode" << std::endl;

    eo::log << eo::setlevel(eo::logging);

    eo::log << eo::errors;
    eo::log << "3) Must be in errors mode";
    eo::log << std::endl;

    eo::log << eo::debug << 4 << ')'
	    << " Must be in debug mode\n";

    return 0;
}

//-----------------------------------------------------------------------------
