//-----------------------------------------------------------------------------
// t-eoLogger.cpp
//-----------------------------------------------------------------------------

#include <eo>

//-----------------------------------------------------------------------------

int main(int ac, char** av)
{
    eoParser parser(ac, av);

    if (parser.userNeedsHelp())
	{
	    parser.printHelp(std::cout);
	    exit(1);
	}

    make_help(parser);
    make_verbose(parser);

    eo::log << eo::setlevel(eo::debug);

    eo::log << eo::warnings;

    eo::log << "We are writing on the default output stream" << std::endl;

    eo::log << eo::file("test.txt") << "In FILE" << std::endl;
    eo::log << std::cout << "on COUT" << std::endl;

    eo::log << eo::setlevel("errors");
    eo::log << eo::setlevel(eo::errors);

    eo::log << eo::quiet << "1) in quiet mode" << std::endl;

    eo::log << eo::setlevel(eo::warnings) << eo::warnings << "2) in warnings mode" << std::endl;

    eo::log << eo::setlevel(eo::logging);

    eo::log << eo::errors;
    eo::log << "3) in errors mode";
    eo::log << std::endl;

    eo::log << eo::debug << 4 << ')'
	    << "4) in debug mode\n";

    return 0;
}

//-----------------------------------------------------------------------------
