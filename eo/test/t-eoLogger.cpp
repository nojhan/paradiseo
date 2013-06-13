//-----------------------------------------------------------------------------
// t-eoLogger.cpp
//-----------------------------------------------------------------------------

#include <eo>
//#include <paradiseo/eo.h>
#include <fstream>
#include <sstream>
#include <string>
#include <cassert>

//-----------------------------------------------------------------------------

static void test();

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

    test();

    return 0;
}

static void test()
{
    eo::log << eo::setlevel(eo::debug);

    eo::log << eo::warnings;

    eo::log << "We are writing on the default output stream" << std::endl;

    {
        eo::log.redirect("logtest.txt");
        eo::log << "In FILE" << std::endl;
        std::ofstream ofs("logtest2.txt"); // closed and destroyed at the en of the scope
        eo::log.addRedirect(ofs);
        eo::log << "In FILE 2" << std::endl;
        eo::log.removeRedirect(ofs);       // must be removed because the associated stream is closed
    }

    std::ifstream ifs("logtest2.txt");     // stream to logtest2.txt is closed, we can start reading
    std::string line;
    assert(std::getline(ifs, line));
    assert(line == "In FILE 2");
    assert(!std::getline(ifs, line));

    std::ostringstream oss;
    eo::log.addRedirect(oss);
    eo::log << "In STRINGSTREAM";
    
    std::cout << "Content of ostringstream: " << oss.str() << std::endl;
    assert(oss.str() == "In STRINGSTREAM");

    eo::log.redirect(std::cout);           // removes all previously redirected streams; closes the file logtest.txt
    eo::log << "on COUT" << std::endl;


    ifs.close();
    ifs.open("logtest.txt");
    assert(std::getline(ifs, line));
    assert(line == "In FILE");
    assert(std::getline(ifs, line));
    assert(line == "In FILE 2");
    assert(std::getline(ifs, line));
    assert(line == "In STRINGSTREAM");
    assert(!std::getline(ifs, line));


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
    
}

//-----------------------------------------------------------------------------
