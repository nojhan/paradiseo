#ifdef _MSC_VER
// to avoid long name warnings
#pragma warning(disable:4786)
#endif 

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <iomanip>
#include <ios>

#include <utils/eoOStreamMonitor.h>
#include <utils/compatibility.h>
#include <utils/eoParam.h>
#include <eoLogger.h>

//using namespace std;

eoMonitor& eoOStreamMonitor::operator()(void)
{
    if (!out) {
        std::string str = "eoOStreamMonitor: Could not write to the ooutput stream";
      throw std::runtime_error(str);
    }

    if (firsttime) {

      if (verbose) {
        eo::log << eo::progress << "First Generation" << std::endl;

      } else { // else verbose
          for (iterator it = vec.begin (); it != vec.end (); ++it) {
              out << (*it)->longName ();
              out << delim << std::left << std::setfill(fill) << std::setw(width);
	      }
          out << std::endl;
	  } // else verbose

      firsttime = false;
    } // if firstime

    // ok, now the real saving. write out
    if (verbose) {
        for (iterator it = vec.begin (); it != vec.end (); ++it) {
            // name: value
            out << (*it)->longName () << ": " << (*it)->getValue () << std::endl;
        } // for it in vec

        eo::log << eo::progress << "End of Generation" << std::endl;

    } else { // else verbose
        for (iterator it = vec.begin (); it != vec.end (); ++it) {
            // value only
            out << (*it)->getValue ();
            out << delim << std::left << std::setfill(fill) << std::setw(width);
        } // for it in vec

        out << std::endl;
    } // if verbose

  return *this;
}
