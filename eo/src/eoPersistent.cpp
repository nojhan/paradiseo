#include <eoPersistent.h>

//Implementation of these objects

//-----------------------------------------------------------------------------

istream & operator >> ( istream& _is, eoPersistent& _o ) {
  _o.readFrom(_is);
  return _is;
};
