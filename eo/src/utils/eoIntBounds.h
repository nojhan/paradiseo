// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoIntBounds.h
// (c) Marc Schoenauer 2001, Maarten Keijzer 2000, GeNeura Team, 1998
/*
    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Contact: todos@geneura.ugr.es, http://geneura.ugr.es
             Marc.Schoenauer@polytechnique.fr
             mak@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef _eoIntBounds_h
#define _eoIntBounds_h

#include <stdexcept>               // std::exceptions!
#include <utils/eoRNG.h>

/**
\class eoIntBounds eoIntBounds.h es/eoIntBounds.h
\ingroup Real
\ingroup Bounds

    Defines bound classes for real numbers.

Scalar type:
------------
Basic class is eoIntBounds, a pure virtual.

The following pure virtual methods are to be used in mutations:
- void foldsInBounds(long int &) that folds any value that falls out of
  the bounds back into the bounds, by bouncing on the limit (if any)
- bool isInBounds(long int) that simply says whether or not the argument
  is in the bounds
- void truncate(long int &) that set the argument to the bound value it
it exceeds it

So mutation can choose
- iterate trying until they fall in bounds,
- only try once and "restd::pair" by using the foldsInBounds method
- only try once and restd::pair using the truncate method (will create a
  huge bias toward the bound if the soluiton is not far from the bounds)

There is also a uniform() method that generates a uniform value
(if possible, i.e. if bounded) in the interval.

Derived class are
eoIntInterval that holds a minimum and maximum value,
eoIntNoBounds the "unbounded bounds" (-infinity, +infinity)
eoIntBelowBound the half-bounded interval [min, +infinity)
eoIntAboveBound the   half-bounded interval (-infinity, max]

THis file also contains the declaration of *the* global object that
is the unbounded bound
*/
class eoIntBounds : public eoPersistent
{
public:
  virtual ~eoIntBounds(){}

  /** Self-Test: true if ***both*** a min and a max
   */
  virtual bool isBounded(void) const  = 0;

  /** Self-Test: true if no min ***and*** no max
   *        hence no further need to test/truncate/fold anything
   */
  virtual bool hasNoBoundAtAll(void) const  = 0;

  /** Self-Test: bounded from below???
   */
  virtual bool isMinBounded(void) const = 0;

  /** Self-Test: bounded from above???
   */
  virtual bool isMaxBounded(void) const = 0;

  /** Test on a value: is it in bounds?
   */
  virtual bool isInBounds(double)  const = 0;

  /** Put value back into bounds - by folding back and forth
   */
  virtual void foldsInBounds(double &)  const = 0;

  /** foldsInBounds for ints:
   * call the method for double and convert back */
  virtual void foldsInBounds(long int & i)  const
  {
    double r = double(i);
    foldsInBounds(r);
    i = (long int)(r);
  }

  /** Put value back into bounds - by truncating to a boundary value
   */
  virtual void truncate(double &)  const = 0;

  /** truncate for ints: call the method for double and convert back
   */
  virtual void truncate(long int & i)  const
  {
    double r = double(i);
    truncate(r);
    i = (long int)(r);
  }

  /** get minimum value
   *  std::exception if does not exist
   */
  virtual long int minimum()  const = 0 ;
  /** get maximum value
   *  std::exception if does not exist
   */
  virtual long int maximum()  const = 0 ;
  /** get range
   *  std::exception if unbounded
   */
  virtual long int range()  const = 0;

  /** random generator of uniform numbers in bounds
   * uses same naming convention than eo::rng
   * std::exception if unbounded
   */
  virtual double uniform(eoRng & _rng = eo::rng)  const = 0;
  virtual long int random(eoRng & _rng = eo::rng)  const = 0;

  /** for memory managements - ugly */
  virtual eoIntBounds * dup()  const = 0;
};

/** A default class for unbounded variables
 *
 * @ingroup Real
 * @ingroup Bounds
 */
class eoIntNoBounds : public eoIntBounds
{
public:
  virtual ~eoIntNoBounds(){}

  virtual bool isBounded(void)  const {return false;}
  virtual bool hasNoBoundAtAll(void) const  {return true;}
  virtual bool isMinBounded(void)  const {return false;}
  virtual bool isMaxBounded(void)  const {return false;}
  virtual void foldsInBounds(double &)  const {return;}
  virtual void truncate(double &)  const {return;}
  virtual bool isInBounds(double)  const {return true;}

  virtual long int minimum() const
  {
    throw std::logic_error("Trying to get minimum of unbounded eoIntBounds");
  }
  virtual long int maximum() const
  {
    throw std::logic_error("Trying to get maximum of unbounded eoIntBounds");
  }
  virtual long int range() const
  {
    throw std::logic_error("Trying to get range of unbounded eoIntBounds");
  }

  virtual double uniform(eoRng & _rng = eo::rng) const
  {
    (void)_rng;

    throw std::logic_error("Trying to generate uniform values in unbounded eoIntBounds");
  }

  virtual long int random(eoRng & _rng = eo::rng) const
  {
    (void)_rng;

    throw std::logic_error("Trying to generate uniform values in unbounded eoIntBounds");
  }

  // methods from eoPersistent
  /**
   * Read object.
   * @param _is A std::istream.
   * but reading should not be done here, because of bound problems
   * see eoIntVectorBounds
   */
  virtual void readFrom(std::istream& _is)
  {
    (void)_is;

    throw std::runtime_error("Should not use eoIntBounds::readFrom");
  }

  /**
   * Write object. It's called printOn since it prints the object on a stream.
   * @param _os A std::ostream.
   */
  virtual void printOn(std::ostream& _os) const
  {
    _os << "[-inf,+inf]";
  }

  /** for memory managements - ugly */
  virtual eoIntBounds * dup() const
  {
    return new eoIntNoBounds(*this);
  }

};

/** one object for all - see eoIntBounds.cpp
 *
 * @ingroup Real
 * @ingroup Bounds
 */
extern eoIntNoBounds eoDummyIntNoBounds;

/**
 * fully bounded eoIntBound == interval
 *
 * @ingroup Real
 * @ingroup Bounds
 */
class eoIntInterval : public eoIntBounds
{
public :
  virtual ~eoIntInterval(){}

  /**
      Simple bounds = minimum and maximum (allowed)
  */
  eoIntInterval(long int _min=0, long int _max=1) :
    repMinimum(_min), repMaximum(_max), repRange(_max-_min)
  {
    if (repRange<=0)
      throw std::logic_error("Void range in eoIntBounds");
  }

  // accessors
  virtual long int minimum() const { return repMinimum; }
  virtual long int maximum() const { return repMaximum; }
  virtual long int range()  const { return repRange; }

  // description
  virtual bool isBounded(void)  const {return true;}
  virtual bool hasNoBoundAtAll(void)  const {return false;}
  virtual bool isMinBounded(void)  const {return true;}
  virtual bool isMaxBounded(void)  const {return true;}

  virtual double uniform(eoRng & _rng = eo::rng) const
  {
    return repMinimum + _rng.uniform(repRange);
  }

  virtual long int random(eoRng & _rng = eo::rng) const
  {
    return repMinimum + _rng.random(repRange);
  }

  // says if a given double is within the bounds
  virtual bool isInBounds(double _r) const
  {
    if (_r < repMinimum)
      return false;
    if (_r > repMaximum)
      return false;
    return true;
  }

  // folds a value into bounds
  virtual void foldsInBounds(double &  _r) const
  {
    long iloc;
    double dlargloc = 2 * range() ;

    if (fabs(_r) > 1.0E9)		// iloc too large!
      {
        _r = uniform();
        return;
      }

    if ( (_r > maximum()) )
      {
        iloc = (long) ( (_r-minimum()) / dlargloc ) ;
        _r -= dlargloc * iloc ;
        if ( _r > maximum() )
          _r = 2*maximum() - _r ;
      }

    if (_r < minimum())
      {
        iloc = (long) ( (maximum()-_r) / dlargloc ) ;
        _r += dlargloc * iloc ;
        if (_r < minimum())
          _r = 2*minimum() - _r ;
      }
  }

  // truncates to the bounds
  virtual void truncate(double & _r) const
  {
    if (_r < repMinimum)
      _r = repMinimum;
    else if (_r > repMaximum)
      _r = repMaximum;
    return;
  }

  // methods from eoPersistent
  /**
   * Read object.
   * @param _is A std::istream.
   * but reading should not be done here, because of bound problems
   * see eoIntVectorBounds
   */
  virtual void readFrom(std::istream& _is)
  {
    (void)_is;

    throw std::runtime_error("Should not use eoIntInterval::readFrom");
  }

  /**
   * Write object. It's called printOn since it prints the object on a stream.
   * @param _os A std::ostream.
   */
  virtual void printOn(std::ostream& _os) const
  {
    _os << "[" << repMinimum << "," << repMaximum << "]";
  }

  /** for memory managements - ugly */
  virtual eoIntBounds * dup() const
  {
    return new eoIntInterval(*this);
  }

private :
  long int repMinimum;
  long int repMaximum;
  long int repRange;                       // to minimize operations ???
};

/**
 * an eoIntBound bounded from below only
 *
 * @ingroup Real
 * @ingroup Bounds
 */
class eoIntBelowBound : public eoIntBounds
{
public :
  virtual ~eoIntBelowBound(){}
  /**
      Simple bounds = minimum
  */
  eoIntBelowBound(long int _min=0) :
    repMinimum(_min)
  {}

  // accessors
  virtual long int minimum() const { return repMinimum; }

  virtual long int maximum() const
  {
    throw std::logic_error("Trying to get maximum of eoIntBelowBound");
  }
  virtual long int range() const
  {
    throw std::logic_error("Trying to get range of eoIntBelowBound");
  }

  virtual double uniform(eoRng & _rng = eo::rng) const
  {
    (void)_rng;

    throw std::logic_error("Trying to generate uniform values in eoIntBelowBound");
  }

  virtual long int random(eoRng & _rng = eo::rng) const
  {
    (void)_rng;

    throw std::logic_error("Trying to generate uniform values in eoIntBelowBound");
  }

  // description
  virtual bool isBounded(void)  const {return false;}
  virtual bool hasNoBoundAtAll(void)  const {return false;}
  virtual bool isMinBounded(void)  const {return true;}
  virtual bool isMaxBounded(void) const  {return false;}

  // says if a given double is within the bounds
  virtual bool isInBounds(double _r) const
  {
    if (_r < repMinimum)
      return false;
    return true;
  }

  // folds a value into bounds
  virtual void foldsInBounds(double &  _r) const
  {
    // easy as a pie: symmetry w.r.t. minimum
    if (_r < repMinimum)           // nothing to do otherwise
      _r = 2*repMinimum - _r;
    return ;
  }

  // truncates to the bounds
  virtual void truncate(double & _r) const
  {
    if (_r < repMinimum)
      _r = repMinimum;
    return;
  }

  // methods from eoPersistent
  /**
   * Read object.
   * @param _is A std::istream.
   * but reading should not be done here, because of bound problems
   * see eoIntVectorBounds
   */
  virtual void readFrom(std::istream& _is)
  {
    (void)_is;

    throw std::runtime_error("Should not use eoIntBelowBound::readFrom");
  }

  /**
   * Write object. It's called printOn since it prints the object on a stream.
   * @param _os A std::ostream.
   */
  virtual void printOn(std::ostream& _os) const
  {
    _os << "[" << repMinimum << ",+inf]";
  }

  /** for memory managements - ugly */
  virtual eoIntBounds * dup() const
  {
    return new eoIntBelowBound(*this);
  }

private :
  long int repMinimum;
};

/**
An eoIntBound bounded from above only
 *
 * @ingroup Real
 * @ingroup Bounds
*/
class eoIntAboveBound : public eoIntBounds
{
public :
  virtual ~eoIntAboveBound(){}

  /**
      Simple bounds = minimum
  */
  eoIntAboveBound(long int _max=0) :
    repMaximum(_max)
  {}

  // accessors
  virtual long int maximum() const  { return repMaximum; }

  virtual long int minimum() const
  {
    throw std::logic_error("Trying to get minimum of eoIntAboveBound");
  }
  virtual long int range() const
  {
    throw std::logic_error("Trying to get range of eoIntAboveBound");
  }

  virtual double uniform(eoRng & _rng = eo::rng) const
  {
    (void)_rng;

    throw std::logic_error("Trying to generate uniform values in eoIntAboveBound");
  }

  virtual long int random(eoRng & _rng = eo::rng) const
  {
    (void)_rng;

    throw std::logic_error("Trying to generate uniform values in eoIntAboveBound");
  }

  // description
  virtual bool isBounded(void)  const {return false;}
  virtual bool hasNoBoundAtAll(void)  const {return false;}
  virtual bool isMinBounded(void)  const {return false;}
  virtual bool isMaxBounded(void)  const {return true;}

  // says if a given double is within the bounds
  virtual bool isInBounds(double _r) const
  {
    if (_r > repMaximum)
      return false;
    return true;
  }

  // folds a value into bounds
  virtual void foldsInBounds(double &  _r) const
  {
    // easy as a pie: symmetry w.r.t. maximum
    if (_r > repMaximum)           // nothing to do otherwise
      _r = 2*repMaximum - _r;
    return ;
  }

  // truncates to the bounds
  virtual void truncate(double & _r) const
  {
    if (_r > repMaximum)
      _r = repMaximum;
    return;
  }

  // methods from eoPersistent
  /**
   * Read object.
   * @param _is A std::istream.
   * but reading should not be done here, because of bound problems
   * see eoIntVectorBounds
   */
  virtual void readFrom(std::istream& _is)
  {
    (void)_is;

    throw std::runtime_error("Should not use eoIntAboveBound::readFrom");
  }

  /**
   * Write object. It's called printOn since it prints the object on a stream.
   * @param _os A std::ostream.
   */
  virtual void printOn(std::ostream& _os) const
  {
    _os << "[-inf," << repMaximum << "]";
  }

  /** for memory managements - ugly */
  virtual eoIntBounds * dup() const
  {
    return new eoIntAboveBound(*this);
  }

private :
  long int repMaximum;
};

//////////////////////// tentative for a general BOUND class that is constructed from a string

/** A class that encapsulate all possible eoIntBounds.
 *  Mandatory in order to read through the parser
 *
 * @ingroup Real
 * @ingroup Bounds
 */
class eoGeneralIntBounds : public eoIntBounds
{
public:
  /** Ctor: from a string, chooses the type of bound */
  eoGeneralIntBounds(std::string _s = "[-infinity,+infinity]")
  {
    repBound = getBoundsFromString(_s);
  }

  /** Need a Cpy Ctor because we are allocating memory */
  eoGeneralIntBounds(const eoGeneralIntBounds & _b) : eoIntBounds(_b)
  {
    // replicate the embedded bound (I'm pretty sure there is another
    // way to do that !!!

    bool minBounded = _b.isMinBounded();
    bool maxBounded = _b.isMaxBounded();
    long int minimum, maximum;
    const eoIntBounds & bb = _b.theBounds();
    if (minBounded) minimum = bb.minimum();
    if (maxBounded) maximum = bb.maximum();

      if (minBounded && maxBounded)
        repBound = new eoIntInterval(minimum, maximum);
      else if (!minBounded && !maxBounded)	// no bound at all
        repBound = new eoIntNoBounds;
      else if (!minBounded && maxBounded)
        repBound = new eoIntAboveBound(maximum);
      else if (minBounded && !maxBounded)
        repBound = new eoIntBelowBound(minimum);
  }

  eoGeneralIntBounds& operator=(const eoGeneralIntBounds& _b)
  {
    // replicate the embedded bound (I'm pretty sure there is another
    // way to do that !!!

    bool minBounded = _b.isMinBounded();
    bool maxBounded = _b.isMaxBounded();
    long int minimum, maximum;
    const eoIntBounds & bb = _b.theBounds();
    if (minBounded) minimum = bb.minimum();
    if (maxBounded) maximum = bb.maximum();

    // first delete the embedded bounds if necessary
    if (repBound)
      delete repBound;
    // now reallocate
      if (minBounded && maxBounded)
        repBound = new eoIntInterval(minimum, maximum);
      else if (!minBounded && !maxBounded)	// no bound at all
        repBound = new eoIntNoBounds;
      else if (!minBounded && maxBounded)
        repBound = new eoIntAboveBound(maximum);
      else if (minBounded && !maxBounded)
        repBound = new eoIntBelowBound(minimum);
      return (*this);
  }


  /** Need a Dtor because we allocate an actual bound  */
  ~eoGeneralIntBounds()
  {
    delete repBound;
  }

  ///// and now all methods from the embedded bounds
  /** Self-Test: true if ***both*** a min and a max
   */
  virtual bool isBounded(void)  const {return repBound->isBounded();}

  /** Self-Test: true if no min ***and*** no max
   *        hence no further need to test/truncate/fold anything
   */
  virtual bool hasNoBoundAtAll(void)  const {return repBound->hasNoBoundAtAll();}

  /** Self-Test: bounded from below???
   */
  virtual bool isMinBounded(void)  const {return repBound->isMinBounded();}

  /** Self-Test: bounded from above???
   */
  virtual bool isMaxBounded(void) const {return repBound->isMaxBounded();}

  /** Test on a value: is it in bounds?
   */
  virtual bool isInBounds(double _x)  const {return repBound->isInBounds(_x);}

  /** Put value back into bounds - by folding back and forth
   */
  virtual void foldsInBounds(double & _x) const {return repBound->foldsInBounds(_x);}

  /** Put value back into bounds - by truncating to a boundary value
   */
  virtual void truncate(double & _x)  const {return repBound->truncate(_x);}

  /** get minimum value
   *  std::exception if does not exist
   */
  virtual long int minimum()  const {return repBound->minimum();}
  /** get maximum value
   *  std::exception if does not exist
   */
  virtual long int maximum() const {return repBound->maximum();}
  /** get range
   *  std::exception if unbounded
   */
  virtual long int range()  const {return repBound->range();}

  /** random generator of uniform doubles in bounds
   * std::exception if unbounded
   */
  virtual double uniform(eoRng & _rng = eo::rng)  const {(void)_rng; return repBound->uniform();}

  /** random generator of uniform ints in bounds
   * std::exception if unbounded
   */
  virtual long int random(eoRng & _rng = eo::rng)  const {(void)_rng; return repBound->random();}

  /** for memory managements - ugly */
  virtual eoIntBounds * dup() const  {return repBound->dup();}

  /** for efficiency, it's better to use the embedded boud directly */
  const eoIntBounds & theBounds()  const { return *repBound;}

  /** don't forget the printOn method -
   * again that of the embedded bound
   */
  virtual void printOn(std::ostream& _os) const
  {
    repBound->printOn(_os);
  }

  /** no readFrom ??? Have to check that later */
  virtual void readFrom(std::istream& _is)
  {
    std::string s;
    _is >> s;
    if (repBound)
      delete repBound;
    repBound = getBoundsFromString(s);
  }

private:
  // reading from a string
  eoIntBounds * getBoundsFromString(std::string);

  eoIntBounds * repBound;
};


#endif
