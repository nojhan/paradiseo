#ifdef _MSC_VER
// to avoid long name warnings
#pragma warning(disable:4786)
#endif

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <ctime>
#include <sstream>

#include "eoRealBounds.h"
#include "eoRealVectorBounds.h"


// the global dummy bounds
// (used for unbounded variables when bounds are required)
eoRealNoBounds eoDummyRealNoBounds;
eoRealVectorNoBounds eoDummyVectorNoBounds(0);

///////////// helper read functions - could be somewhere else

// removes leading delimiters - return false if nothing else left
bool remove_leading(std::string & _s, const std::string _delim)
{
  size_t posDebToken = _s.find_first_not_of(_delim);
  if (posDebToken >= _s.size())
    return false;
  _s = _s.substr(posDebToken);
  return true;
}

double read_double(std::string _s)
{
    std::istringstream is(_s);
  double r;
  is >> r;
  return r;
}

long int read_int(std::string _s)
{
    std::istringstream is(_s);
  long int i;
  is >> i;
  return i;
}

// need to rewrite copy ctor and assignement operator because of ownedBounds
eoRealVectorBounds::eoRealVectorBounds(const eoRealVectorBounds & _b):
    eoRealBaseVectorBounds(_b), eoPersistent()
{
  factor = _b.factor;
  ownedBounds = _b.ownedBounds;
  // duplicate all pointers!
  if (ownedBounds.size()>0)
    for (unsigned i=0; i<ownedBounds.size(); i++)
      ownedBounds[i] = ownedBounds[i]->dup();
}


// the readFrom method of eoRealVectorNoBounds:
// only calls the readFrom(string) - for param reading
void eoRealVectorBounds::readFrom(std::istream& _is)
{
  std::string value;
  _is >> value;
  readFrom(value);
  return;
}

void eoRealVectorBounds::readFrom(std::string _value)
{
  // keep track of old size - to adjust in the end
  unsigned oldSize = size();
  // clean-up before filling in
  if (ownedBounds.size()>0)
    for (unsigned i = 0; i < ownedBounds.size(); ++i)
      {
        delete ownedBounds[i];
      }
  ownedBounds.resize(0);
  factor.resize(0);
  resize(0);

  // now read
  std::string delim(",; ");
  while (_value.size()>0)
    {
      if (!remove_leading(_value, delim)) // only delimiters were left
        break;
      // look for opening char
      size_t posDeb = _value.find_first_of("[(");
      if (posDeb >= _value.size())	// nothing left to read (though probably a syntax error there)
        {
          break;
        }
      // ending char
      std::string closeChar = (_value[posDeb] == '(' ? std::string(")") : std::string("]") );

      size_t posFin = _value.find_first_of(std::string(closeChar));
      if (posFin >= _value.size())
        throw std::runtime_error("Syntax error when reading bounds");

  // y a-t-il un nbre devant
      unsigned count = 1;
      if (posDeb > 0)			// something before opening
        {
          std::string sCount = _value.substr(0, posDeb);
          count = read_int(sCount);
          if (count <= 0)
            throw std::runtime_error("Syntax error when reading bounds");
        }

      // the bounds
      std::string sBounds = _value.substr(posDeb+1, posFin-posDeb-1);
      // and remove from original string
      _value = _value.substr(posFin+1);

      remove_leading(sBounds, delim);
      size_t posDelim = sBounds.find_first_of(delim);
      if (posDelim >= sBounds.size())
        throw std::runtime_error("Syntax error when reading bounds");

      bool minBounded=false, maxBounded=false;
      double minBound=0, maxBound=0;

      // min bound
      std::string sMinBounds = sBounds.substr(0,posDelim);
      if (sMinBounds != std::string("-inf"))
        {
          minBounded = true;
          minBound = read_double(sMinBounds);
        }

      // max bound
      size_t posEndDelim = sBounds.find_first_not_of(delim,posDelim);

      std::string sMaxBounds = sBounds.substr(posEndDelim);
      if (sMaxBounds != std::string("+inf"))
        {
          maxBounded = true;
          maxBound = read_double(sMaxBounds);
        }

      // now create the eoRealBounds objects
      eoRealBounds *ptBounds;
      if (minBounded && maxBounded)
        ptBounds = new eoRealInterval(minBound, maxBound);
      else if (!minBounded && !maxBounded)	// no bound at all
        ptBounds = new eoRealNoBounds;
      else if (!minBounded && maxBounded)
        ptBounds = new eoRealAboveBound(maxBound);
      else if (minBounded && !maxBounded)
        ptBounds = new eoRealBelowBound(minBound);
      // store it for memory management
      ownedBounds.push_back(ptBounds);
      // push the count
      factor.push_back(count);
      // and add count of it to the actual bounds
      for (unsigned i=0; i<count; i++)
        push_back(ptBounds);
    }
  // now adjust the size to the initial value
  adjust_size(oldSize);
}

/** Eventually increases the size by duplicating last bound */
void eoRealVectorBounds::adjust_size(unsigned _dim)
{
  if ( size() < _dim )
    {
      // duplicate last bound
      unsigned missing = _dim-size();
      eoRealBounds * ptBounds = back();
      for (unsigned i=0; i<missing; i++)
        push_back(ptBounds);
      // update last factor (warning: can be > 1 already!)
      factor[factor.size()-1] += missing;
    }
}

/** the constructor for eoGeneralRealBound - from a string
 *  very similar to the eoRealVectorBounds::readFrom above
 *  but was written much later so the readFrom does not call this one
 *  as it should do
 */
eoRealBounds* eoGeneralRealBounds::getBoundsFromString(std::string _value)
{
  // now read
  std::string delim(",; ");
  std::string beginOrClose("[(])");
  if (!remove_leading(_value, delim)) // only delimiters were left
    throw std::runtime_error("Syntax error in eoGeneralRealBounds Ctor");

  // look for opening char
  size_t posDeb = _value.find_first_of(beginOrClose);	// allow ]a,b]
  if (posDeb >= _value.size())	// nothing left to read
    throw std::runtime_error("Syntax error in eoGeneralRealBounds Ctor");

  // ending char: next {}() after posDeb
  size_t posFin = _value.find_first_of(beginOrClose,posDeb+1);
  if (posFin >= _value.size())	// not found
    throw std::runtime_error("Syntax error in eoGeneralRealBounds Ctor");

  // the bounds
  std::string sBounds = _value.substr(posDeb+1, posFin-posDeb-1);
  // and remove from original string
  _value = _value.substr(posFin+1);

  remove_leading(sBounds, delim);
  size_t posDelim = sBounds.find_first_of(delim);
  if (posDelim >= sBounds.size())
    throw std::runtime_error("Syntax error in eoGeneralRealBounds Ctor");

      bool minBounded=false, maxBounded=false;
      double minBound=0, maxBound=0;

      // min bound
      std::string sMinBounds = sBounds.substr(0,posDelim);

      if ( (sMinBounds != std::string("-inf")) &&
           (sMinBounds != std::string("-infinity"))
           )
        {
          minBounded = true;
          minBound = read_double(sMinBounds);
        }

      // max bound
      size_t posEndDelim = sBounds.find_first_not_of(delim,posDelim);

      std::string sMaxBounds = sBounds.substr(posEndDelim);

      if ( (sMaxBounds != std::string("+inf")) &&
           (sMaxBounds != std::string("+infinity"))
           )
        {
          maxBounded = true;
          maxBound = read_double(sMaxBounds);
        }

      // now create the embedded eoRealBounds object
      eoRealBounds * locBound;
      if (minBounded && maxBounded)
        {
          if (maxBound <= minBound)
            throw std::runtime_error("Syntax error in eoGeneralRealBounds Ctor");
          locBound = new eoRealInterval(minBound, maxBound);
        }
      else if (!minBounded && !maxBounded)	// no bound at all
        locBound = new eoRealNoBounds;
      else if (!minBounded && maxBounded)
        locBound = new eoRealAboveBound(maxBound);
      else if (minBounded && !maxBounded)
        locBound = new eoRealBelowBound(minBound);
      return locBound;
}
