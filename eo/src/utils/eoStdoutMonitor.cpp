#include <iostream>
#include <fstream>
#include <stdexcept>

#include <utils/eoStdoutMonitor.h>
#include <utils/compatibility.h>
#include <utils/eoParam.h>

using namespace std;

eoMonitor& eoStdoutMonitor::operator()(void)
{
    if (firsttime)
    {
        cout << "First Generation" << endl;

        firsttime = false;

    }
    // ok, now the real saving. write out

    if (!cout)
    {
        string str = "eoStdoutMonitor: Could not write to cout";
        throw runtime_error(str);
    }

    for(iterator it = vec.begin(); it != vec.end(); ++it)
    {
        cout << (*it)->longName() << ": " << (*it)->getValue() << '\n';
    }

    cout << "\n****** End of Generation ******\n\n"; 

    return *this;
}

