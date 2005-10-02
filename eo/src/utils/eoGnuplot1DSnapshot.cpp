#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "eoGnuplot1DSnapshot.h"



inline eoMonitor&   eoGnuplot1DSnapshot::operator() (void)
{
  // update file using the eoFileMonitor method
  eoFileSnapshot::operator()();

  // sends plot order to gnuplot
  //std::string buff; // need local memory
  std::ostringstream os;
  os << "set title 'Gen. " << getCounter() << "'; plot '"
    // mk: had to use getFilename().c_str(), because it seems the string(stream) lib is screwed in gcc3.2
      << getFileName().c_str() << "' notitle with points ps " << pointSize;
  os << std::endl;
  PipeComSend( gpCom, os.str().c_str());
  return (*this);
}



// Local Variables:
// c-file-style: "Stroustrup"
// fill-column: 80
// End:
