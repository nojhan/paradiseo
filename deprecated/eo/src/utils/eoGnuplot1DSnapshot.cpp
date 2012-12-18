#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "eoGnuplot1DSnapshot.h"



eoMonitor& eoGnuplot1DSnapshot::operator()()
{
    // update file using the eoFileMonitor method
    eoFileSnapshot::operator()();
#ifdef HAVE_GNUPLOT
    // sends plot order to gnuplot
    std::ostringstream os;
    os << "set title 'Gen. " << getCounter() << "'; plot '"
        // mk: had to use getFilename().c_str(),
        // because it seems the string(stream) lib is screwed in gcc3.2
       << getFileName().c_str() << "' notitle with points ps " << pointSize
       << std::endl;
    PipeComSend(gpCom, os.str().c_str());
#endif
    return *this;
}


// Local Variables:
// coding: iso-8859-1
// mode: C++
// c-file-offsets: ((c . 0))
// c-file-style: "Stroustrup"
// fill-column: 80
// End:
