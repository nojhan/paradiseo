#ifdef _MSC_VER
// to avoid long name warnings
#pragma warning(disable:4786)
#endif 

#include <iostream>
#include <fstream>
#include <stdexcept>

#include <utils/eoStdoutMonitor.h>
#include <utils/compatibility.h>
#include <utils/eoParam.h>
#include <eoLogger.h>

using namespace std;

eoMonitor& eoStdoutMonitor::operator()(void)
{
    if (!cout)
    {
        string str = "eoStdoutMonitor: Could not write to cout";
        throw runtime_error(str);
    }
    if (firsttime)
    {
      if (verbose)
        eo::log << eo::progress << "First Generation" << endl;
      else
	{
	  for(iterator it = vec.begin(); it != vec.end(); ++it)
	    {
	      cout << (*it)->longName() << delim;
	    }
	  cout << endl;
	}
        firsttime = false;
    }
    // ok, now the real saving. write out
    if (verbose)
      {
      for(iterator it = vec.begin(); it != vec.end(); ++it)
	{
	  cout << (*it)->longName() << ": " << (*it)->getValue() << endl;
	}
      eo::log << eo::progress << "End of Generation" << endl;
      }
    else			// a one-liner
      {
	for(iterator it = vec.begin(); it != vec.end(); ++it)
	  {
	    cout << (*it)->getValue() << delim;
	  }
	cout << endl;
      }
    return *this;
}
