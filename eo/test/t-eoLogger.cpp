//-----------------------------------------------------------------------------
// t-eoLogger.cpp
//-----------------------------------------------------------------------------

#include <eo>
//#include <paradiseo/eo.h>
#include <fstream>
#include <sstream>
#include <string>
#include <cassert>

// TODO test multiple redirection

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

    {
        std::ofstream ofs("logtest.txt");
        eo::log.redirect(ofs);
        eo::log << "In FILE" << std::endl;
        eo::log.redirect("logtest2.txt");
        eo::log << "In FILE 2" << std::endl;
    }

    std::ifstream ifs("logtest.txt");
    //ifs >> str;
    std::string line;
    assert(std::getline(ifs, line));
    assert(line == "In FILE");
    //std::cout << (line == "In FILE") << std::endl;
    assert(!std::getline(ifs, line));

    std::ostringstream oss;
    eo::log.redirect(oss);
    //eo::log << oss << "In STRINGSTREAM";
    eo::log << "In STRINGSTREAM";
    
    std::cout << "Content of ostringstream: " << oss.str() << std::endl;
    assert(oss.str() == "In STRINGSTREAM");

    ifs.close();
    ifs.open("logtest2.txt");
    assert(std::getline(ifs, line));
    assert(line == "In FILE 2");
    assert(!std::getline(ifs, line));
    //assert(false);


    eo::log.redirect(std::cout);
    eo::log << "on COUT" << std::endl;

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
