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
  virtual bool operator<(const eoFitness& other) const = 0;

  bool operator>(const eoFitness& other) const
    {
      return !(*this < other || *this == other);
    }

  bool operator==(const eoFitness& other) const
    {
      return !(other < *this || *this < other);
    }

  bool operator!=(const eoFitness& other) const
    {
      return other < *this || *this < other;
    }

  virtual operator float() const = 0;
};

//-----------------------------------------------------------------------------

#endif EOFITNESS_H
