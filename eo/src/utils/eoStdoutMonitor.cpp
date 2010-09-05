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
    if (!cout) {
      string str = "eoStdoutMonitor: Could not write to cout";
      throw runtime_error (str);
    }

    if (firsttime) {

      if (verbose) {
        eo::log << eo::progress << "First Generation" << endl;

      } else { // else verbose
          for (iterator it = vec.begin (); it != vec.end (); ++it) {
              cout << (*it)->longName () << delim;
	      }
          cout << endl;
	  } // else verbose

      firsttime = false;
    } // if firstime

    // ok, now the real saving. write out
    if (verbose) {
        for (iterator it = vec.begin (); it != vec.end (); ++it) {
            // name: value
            cout << (*it)->longName () << ": " << (*it)->getValue () << endl;
        } // for it in vec

        eo::log << eo::progress << "End of Generation" << endl;

    } else { // else verbose
        for (iterator it = vec.begin (); it != vec.end (); ++it) {
            // value only
            cout << (*it)->getValue () << delim;
        } // for it in vec

        cout << endl;
    } // if verbose

  return *this;
}
