#include <iostream>
#include <fstream>
#include <stdexcept>

#include <utils/eoFileMonitor.h>
#include <utils/compatibility.h>
#include <utils/eoParam.h>

using namespace std;

eoMonitor& eoFileMonitor::operator()(void)
{
    if (firsttime)
    {
        firsttime = false;

        // create file
        ofstream os(filename.c_str()); 

        if (!os)
        {
            string str = "eoFileMonitor: Could not open " + filename;
            throw runtime_error(str);
        }

        iterator it = vec.begin();

        os << (*it)->longName();

        ++it;

        for (; it != vec.end(); ++it)
        {
            os << ',' << (*it)->longName();
        }
    }
    // ok, now the real saving. append to file

    ofstream os(filename.c_str(), ios_base::app);

    if (!os)
    {
        string str = "eoFileMonitor: Could not append to " + filename;
        throw runtime_error(str);
    }

    iterator it = vec.begin();
    
    os << '\n' << (*it)->getValue();

    for(++it; it != vec.end(); ++it)
    {
        os << ',' << (*it)->getValue();
    }

    return *this;
}

