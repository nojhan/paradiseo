//-----------------------------------------------------------------------------
// eoFitness.cpp
// (c) GeNeura Team 1998
//-----------------------------------------------------------------------------

#ifndef EOFITNESS_H
#define EOFITNESS_H

//-----------------------------------------------------------------------------

class eoFitness: public eoPersistent
{
 public:
  virtual operator float() const = 0;
  
  virtual bool operator<(const eoFitness& other) const = 0;
  
  virtual bool operator>(const eoFitness& other) const
    {
      return !(*this < other || *this == other);
    }
  
  virtual bool operator==(const eoFitness& other) const
    {
      return !(other < *this || *this < other);
    }

  virtual bool operator!=(const eoFitness& other) const
    {
      return other < *this || *this < other;
    }
};

//-----------------------------------------------------------------------------

#endif EOFITNESS_H
