#ifdef _MSC_VER
// to avoid long name warnings
#pragma warning(disable:4786)
#endif

#include <eoPersistent.h>

//Implementation of these objects


std::istream & operator >> ( std::istream& _is, eoPersistent& _o ) {
  _o.readFrom(_is);
  return _is;
}
