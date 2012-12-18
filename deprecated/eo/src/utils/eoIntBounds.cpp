#ifdef _MSC_VER
// to avoid long name warnings
#pragma warning(disable:4786)
#endif

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <ctime>
#include <sstream>

#include "eoIntBounds.h"


// the global dummy bounds
// (used for unbounded variables when bounds are required)
eoIntNoBounds eoDummyIntNoBounds;

///////////// helper read functions defined in eoRealBounds.cpp
extern bool remove_leading(std::string & _s, const std::string _delim);
extern double read_double(std::string _s);
extern long int read_int(std::string _s);


/** the constructor for eoGeneralIntBound - from a string
 */
eoIntBounds* eoGeneralIntBounds::getBoundsFromString(std::string _value)
{
  // now read
  std::string delim(",; ");
  std::string beginOrClose("[(])");
  if (!remove_leading(_value, delim)) // only delimiters were left
    throw std::runtime_error("Syntax error in eoGeneralIntBounds Ctor");

  // look for opening char
  size_t posDeb = _value.find_first_of(beginOrClose);	// allow ]a,b]
  if (posDeb >= _value.size())	// nothing left to read
    throw std::runtime_error("Syntax error in eoGeneralIntBounds Ctor");

  // ending char: next {}() after posDeb
  size_t posFin = _value.find_first_of(beginOrClose,posDeb+1);
  if (posFin >= _value.size())	// not found
    throw std::runtime_error("Syntax error in eoGeneralIntBounds Ctor");

  // the bounds
  std::string sBounds = _value.substr(posDeb+1, posFin-posDeb-1);
  // and remove from original string
  _value = _value.substr(posFin+1);

  remove_leading(sBounds, delim);
  size_t posDelim = sBounds.find_first_of(delim);
  if (posDelim >= sBounds.size())
    throw std::runtime_error("Syntax error in eoGeneralIntBounds Ctor");

      bool minBounded=false, maxBounded=false;
      long int minBound=0, maxBound=0;

      // min bound
      std::string sMinBounds = sBounds.substr(0,posDelim);

      if ( (sMinBounds != std::string("-inf")) &&
           (sMinBounds != std::string("-infinity"))
           )
        {
          minBounded = true;
          minBound = read_int(sMinBounds);
        }

      // max bound
      size_t posEndDelim = sBounds.find_first_not_of(delim,posDelim);

      std::string sMaxBounds = sBounds.substr(posEndDelim);

      if ( (sMaxBounds != std::string("+inf")) &&
           (sMaxBounds != std::string("+infinity"))
           )
        {
          maxBounded = true;
          maxBound = read_int(sMaxBounds);
        }

      // now create the embedded eoIntBounds object
      eoIntBounds * locBound;
      if (minBounded && maxBounded)
        {
          if (maxBound <= minBound)
            throw std::runtime_error("Syntax error in eoGeneralIntBounds Ctor");
          locBound = new eoIntInterval(minBound, maxBound);
        }
      else if (!minBounded && !maxBounded)	// no bound at all
        locBound = new eoIntNoBounds;
      else if (!minBounded && maxBounded)
        locBound = new eoIntAboveBound(maxBound);
      else if (minBounded && !maxBounded)
        locBound = new eoIntBelowBound(minBound);
      return locBound;
}
