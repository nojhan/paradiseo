#ifdef _MSC_VER
// to avoid long name warnings
#pragma warning(disable:4786)
#endif 

#include <iostream>
#include <fstream>
#include <stdexcept>

#include <utils/eoFileMonitor.h>
#include <utils/compatibility.h>
#include <utils/eoParam.h>

using namespace std;

void eoFileMonitor::printHeader(std::ostream& os)
{
    iterator it = vec.begin();

    os << (*it)->longName();

    ++it;

    for (; it != vec.end(); ++it)
    {
        os << delim.c_str() << (*it)->longName();
    }
    os << '\n';
}

void eoFileMonitor::printHeader()
{
    // create file
    ofstream os(filename.c_str()); 

    if (!os)
    {
        string str = "eoFileMonitor: Could not open " + filename;
        throw runtime_error(str);
    }
    
    printHeader(os);
}

eoMonitor& eoFileMonitor::operator()(void)
{
    ofstream os(filename.c_str(), ios_base::app);

    if (!os)
    {
        string str = "eoFileMonitor: Could not append to " + filename;
        throw runtime_error(str);
    }

    if (firstcall && !keep && header ){
      printHeader();
      firstcall = false;
    }
    
    return operator()(os);
}

eoMonitor& eoFileMonitor::operator()(std::ostream& os)
{
    iterator it = vec.begin();

    os << (*it)->getValue();
    
    for(++it; it != vec.end(); ++it)
    {
        os << delim.c_str() << (*it)->getValue();
    }

    os << '\n';
    return *this;
}

