// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoRealBounds.h
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

#ifndef _eoRealBounds_h
#define _eoRealBounds_h

#include <stdexcept>		   // std::exceptions!
#include <utils/eoRNG.h>

/**
\defgroup EvolutionStrategies

*/


/**
\class eoRealBounds eoRealBounds.h es/eoRealBounds.h
\ingroup EvolutionStrategies

    Defines bound classes for real numbers.

Scalar type:
------------
Basic class is eoRealBounds, a pure virtual.

The following pure virtual methods are to be used in mutations:
- void foldsInBounds(double &) that folds any value that falls out of 
  the bounds back into the bounds, by bouncing on the limit (if any)
- bool isInBounds(double) that simply says whether or not the argument 
  is in the bounds
- void truncate(double &) that set the argument to the bound value it
it exceeds it

So mutation can choose 
- iterate trying until they fall in bounds, 
- only try once and "restd::pair" by using the foldsInBounds method
- only try once and restd::pair using the truncate method (will create a
  huge bias toward the bound if the soluiton is not far from the bounds)

There is also a uniform() method that generates a uniform value 
(if possible, i.e. if bounded) in the interval.

Derived class are 
eoRealInterval that holds a minimum and maximum value, 
eoRealNoBounds the "unbounded bounds" (-infinity, +infinity)
eoRealBelowBound the half-bounded interval [min, +infinity)
eoRealAboveBound the   half-bounded interval (-infinity, max]

THis file also contains the declaration of *the* global object that
is the unbounded bound
*/
class eoRealBounds : public eoPersistent
{ 
public:
  virtual ~eoRealBounds(){}

  /** Self-Test: true if ***both*** a min and a max
   */
  virtual bool isBounded(void) = 0;

  /** Self-Test: true if no min ***and*** no max
   *        hence no further need to test/truncate/fold anything
   */
  virtual bool hasNoBoundAtAll(void) = 0;

  /** Self-Test: bounded from below???
   */
  virtual bool isMinBounded(void) = 0;

  /** Self-Test: bounded from above???
   */
  virtual bool isMaxBounded(void) = 0;

  /** Test on a value: is it in bounds?
   */
  virtual bool isInBounds(double) = 0;

  /** Put value back into bounds - by folding back and forth
   */
  virtual void foldsInBounds(double &) = 0;

  /** Put value back into bounds - by truncating to a boundary value
   */
  virtual void truncate(double &) = 0;

  /** get minimum value 
   *  @std::exception if does not exist
   */  
  virtual double minimum() = 0;
  /** get maximum value
   *  @std::exception if does not exist
   */  
  virtual double maximum() = 0;
  /** get range
   *  @std::exception if unbounded
   */  
  virtual double range() = 0;

  /** random generator of uniform numbers in bounds
   * @std::exception if unbounded
   */
  virtual double uniform(eoRng & _rng = eo::rng) = 0;

  /** for memory managements - ugly */
  virtual eoRealBounds * dup() = 0;
};

/** A default class for unbounded variables
 */
class eoRealNoBounds : public eoRealBounds
{
public:
  virtual ~eoRealNoBounds(){}

  virtual bool isBounded(void) {return false;}
  virtual bool hasNoBoundAtAll(void) {return true;}
  virtual bool isMinBounded(void) {return false;}
  virtual bool isMaxBounded(void) {return false;}
  virtual void foldsInBounds(double &) {return;}
  virtual void truncate(double &) {return;}
  virtual bool isInBounds(double) {return true;}

  virtual double minimum()
  {
    throw std::logic_error("Trying to get minimum of unbounded eoRealBounds");
  }
  virtual double maximum()
  {
    throw std::logic_error("Trying to get maximum of unbounded eoRealBounds");
  }
  virtual double range()
  {
    throw std::logic_error("Trying to get range of unbounded eoRealBounds");
  }

  virtual double uniform(eoRng & _rng = eo::rng)
  {
    throw std::logic_error("Trying to generate uniform values in unbounded eoRealBounds");
  }

  // methods from eoPersistent
  /**
   * Read object.
   * @param _is A std::istream.
   * but reading should not be done here, because of bound problems
   * see eoRealVectorBounds
   */
  virtual void readFrom(std::istream& _is) 
  {
    throw std::runtime_error("Should not use eoRealBounds::readFrom");
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
  virtual eoRealBounds * dup()
  {
    return new eoRealNoBounds(*this);
  }

};

// one object for all - see eoRealBounds.cpp
extern eoRealNoBounds eoDummyRealNoBounds;

/**
 * fully bounded eoRealBound == interval
 */
class eoRealInterval : public eoRealBounds
{
public :
  virtual ~eoRealInterval(){}
  
  /** 
      Simple bounds = minimum and maximum (allowed)
  */
  eoRealInterval(double _min=0, double _max=1) : 
    repMinimum(_min), repMaximum(_max), repRange(_max-_min) 
  {
    if (repRange<=0)
      throw std::logic_error("Void range in eoRealBounds");
  }

  // accessors  
  virtual double minimum() { return repMinimum; }
  virtual double maximum() { return repMaximum; }
  virtual double range()   { return repRange; }

  // description
  virtual bool isBounded(void) {return true;}
  virtual bool hasNoBoundAtAll(void) {return false;}
  virtual bool isMinBounded(void) {return true;}
  virtual bool isMaxBounded(void) {return true;}

  virtual double uniform(eoRng & _rng = eo::rng)
  {
    return repMinimum + _rng.uniform(repRange);
  }  

  // says if a given double is within the bounds
  virtual bool isInBounds(double _r)
  {
    if (_r < repMinimum)
      return false;
    if (_r > repMaximum)
      return false;
    return true;
  }

  // folds a value into bounds
  virtual void foldsInBounds(double &  _r)
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
  virtual void truncate(double & _r)
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
   * see eoRealVectorBounds
   */
  virtual void readFrom(std::istream& _is) 
  {
    throw std::runtime_error("Should not use eoRealInterval::readFrom");
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
  virtual eoRealBounds * dup()
  {
    return new eoRealInterval(*this);
  }

private :
  double repMinimum;
  double repMaximum;
  double repRange;			   // to minimize operations ???
};

/**
 * an eoRealBound bounded from below only
 */
class eoRealBelowBound : public eoRealBounds
{
public :
  virtual ~eoRealBelowBound(){}  
  /** 
      Simple bounds = minimum
  */
  eoRealBelowBound(double _min=0) : 
    repMinimum(_min)
  {}

  // accessors  
  virtual double minimum() { return repMinimum; }

  virtual double maximum()
  {
    throw std::logic_error("Trying to get maximum of eoRealBelowBound");
  }
  virtual double range()
  {
    throw std::logic_error("Trying to get range of eoRealBelowBound");
  }

  // random generators
  virtual double uniform(eoRng & _rng = eo::rng)
  {
    throw std::logic_error("Trying to generate uniform values in eoRealBelowBound");
  }

  // description
  virtual bool isBounded(void) {return false;}
  virtual bool hasNoBoundAtAll(void) {return false;}
  virtual bool isMinBounded(void) {return true;}
  virtual bool isMaxBounded(void) {return false;}

  // says if a given double is within the bounds
  virtual bool isInBounds(double _r)
  {
    if (_r < repMinimum)
      return false;
    return true;
  }

  // folds a value into bounds
  virtual void foldsInBounds(double &  _r)
  {
    // easy as a pie: symmetry w.r.t. minimum
    if (_r < repMinimum)	   // nothing to do otherwise
      _r = 2*repMinimum - _r;
    return ;
  }    

  // truncates to the bounds
  virtual void truncate(double & _r)
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
   * see eoRealVectorBounds
   */
  virtual void readFrom(std::istream& _is) 
  {
    throw std::runtime_error("Should not use eoRealBelowBound::readFrom");
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
  virtual eoRealBounds * dup()
  {
    return new eoRealBelowBound(*this);
  }

private :
  double repMinimum;
};

/**
An eoRealBound bounded from above only
*/
class eoRealAboveBound : public eoRealBounds
{
public :
  virtual ~eoRealAboveBound(){}
  
  /** 
      Simple bounds = minimum
  */
  eoRealAboveBound(double _max=0) : 
    repMaximum(_max)
  {}

  // accessors  
  virtual double maximum() { return repMaximum; }

  virtual double minimum()
  {
    throw std::logic_error("Trying to get minimum of eoRealAboveBound");
  }
  virtual double range()
  {
    throw std::logic_error("Trying to get range of eoRealAboveBound");
  }

  // random generators
  virtual double uniform(eoRng & _rng = eo::rng)
  {
    throw std::logic_error("Trying to generate uniform values in eoRealAboveBound");
  }

  // description
  virtual bool isBounded(void) {return false;}
  virtual bool hasNoBoundAtAll(void) {return false;}
  virtual bool isMinBounded(void) {return false;}
  virtual bool isMaxBounded(void) {return true;}

  // says if a given double is within the bounds
  virtual bool isInBounds(double _r)
  {
    if (_r > repMaximum)
      return false;
    return true;
  }

  // folds a value into bounds
  virtual void foldsInBounds(double &  _r)
  {
    // easy as a pie: symmetry w.r.t. maximum
    if (_r > repMaximum)	   // nothing to do otherwise
      _r = 2*repMaximum - _r;
    return ;
  }    

  // truncates to the bounds
  virtual void truncate(double & _r)
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
   * see eoRealVectorBounds
   */
  virtual void readFrom(std::istream& _is) 
  {
    throw std::runtime_error("Should not use eoRealAboveBound::readFrom");
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
  virtual eoRealBounds * dup()
  {
    return new eoRealAboveBound(*this);
  }

private :
  double repMaximum;
};

#endif
