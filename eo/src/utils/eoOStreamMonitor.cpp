#ifdef _MSC_VER
// to avoid long name warnings
#pragma warning(disable:4786)
#endif

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <iomanip>
#include <ios>

#include "eoOStreamMonitor.h"
#include "compatibility.h"
#include "eoParam.h"
#include "eoLogger.h"

//using namespace std;

eoMonitor& eoOStreamMonitor::operator()(void)
{
    if (!out) {
        // std::string str = "eoOStreamMonitor: Could not write to the output stream";
      throw eoFileError("output stream");
    }

    if (firsttime) {

        eo::log << eo::debug << "First Generation" << std::endl;

        for (iterator it = vec.begin (); it != vec.end (); ++it) {
            out << (*it)->longName ();
            out << delim << std::left << std::setfill(fill) << std::setw(width);
        }
        out << std::endl;

        firsttime = false;
    } // if firstime

    // ok, now the real saving. write out
    // FIXME deprecated, remove in next release
    //! @todo old verbose formatting, do we still need it?
    /*
        for (iterator it = vec.begin (); it != vec.end (); ++it) {
            // name: value
            out << (*it)->longName () << ": " << (*it)->getValue () << std::endl;
        } // for it in vec
    */

    for (iterator it = vec.begin (); it != vec.end (); ++it) {
        if( print_names ) {
            out << (*it)->longName() << name_sep;
        }
        out << (*it)->getValue();
        out << delim << std::left << std::setfill(fill) << std::setw(width);
    } // for it in vec

    out << std::endl;
    eo::log << eo::debug << "End of Generation" << std::endl;

  return *this;
}
