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

#include <stdexcept>		   // exceptions!
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
- only try once and "repair" by using the foldsInBounds method
- only try once and repair using the truncate method (will create a
  huge bias toward the bound if the soluiton is not far from the bounds)

There is also a uniform() method that generates a uniform value 
(if possible, i.e. if bounded) in the interval.

Derived class are 
eoRealInterval that holds a minimum and maximum value, 
eoRealNoBounds the "unbounded bounds" (-infinity, +infinity)
eoRealBelowBound the half-bounded interval [min, +infinity)
eoRealAboveBound the   half-bounded interval (-infinity, max]

Vector type: 
------------
Class eoRealVectorBounds implements the vectorized version: 
it is basically a vector of eoRealBounds * and forwards all request
to the elements of the vector.

This file also contains te 2 global variables eoDummyRealNoBounds and
eoDummyVectorNoBounds that are used as defaults in ctors (i.e. when no
bounds are given, it is assumed unbounded values)

TODO: have an eoRealBounds.cpp with the longuish parts of the code
(and the 2 global variables).
*/
class eoRealBounds
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
   *  @exception if does not exist
   */  
  virtual double minimum() = 0;
  /** get maximum value
   *  @exception if does not exist
   */  
  virtual double maximum() = 0;
  /** get range
   *  @exception if unbounded
   */  
  virtual double range() = 0;

  /** random generator of uniform numbers in bounds
   * @exception if unbounded
   */
  virtual double uniform(eoRng & _rng = eo::rng) = 0;
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
    throw logic_error("Trying to get minimum of unbounded eoRealBounds");
  }
  virtual double maximum()
  {
    throw logic_error("Trying to get maximum of unbounded eoRealBounds");
  }
  virtual double range()
  {
    throw logic_error("Trying to get range of unbounded eoRealBounds");
  }

  virtual double uniform(eoRng & _rng = eo::rng)
  {
    throw logic_error("Trying to generate uniform values in unbounded eoRealBounds");
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
    throw logic_error("Trying to get maximum of eoRealBelowBound");
  }
  virtual double range()
  {
    throw logic_error("Trying to get range of eoRealBelowBound");
  }

  // random generators
  virtual double uniform(eoRng & _rng = eo::rng)
  {
    throw logic_error("Trying to generate uniform values in eoRealBelowBound");
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
    throw logic_error("Trying to get minimum of eoRealAboveBound");
  }
  virtual double range()
  {
    throw logic_error("Trying to get range of eoRealAboveBound");
  }

  // random generators
  virtual double uniform(eoRng & _rng = eo::rng)
  {
    throw logic_error("Trying to generate uniform values in eoRealAboveBound");
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

private :
  double repMaximum;
};

/////////////////////////////////////////////////////////////////////
// The Vectorized versions
/////////////////////////////////////////////////////////////////////

/** 
Class eoRealVectorBounds implements the vectorized version: 
it is basically a vector of eoRealBounds * and forwards all request
to the elements of the vector.
Probably it would have been cleaner if there had been an empty base class
from which eoRealVectorBounds AND eoRealVectorNoBounds would have derived.
This is because I started to write eoRealVectorNoBounds as a 
   vector<eoRealBounds *>  whose compoenents would have been eoRealNoBounds
   but then realize that you don't necessarily have the dimension 
   when construction this vector - hence I added the eoRealVectorNoBounds ...
Anyone with extra time in his agenda is welcome to change that :-)
*/
class eoRealVectorBounds : public vector<eoRealBounds *>
{ 
public:
  // virtual desctructor (to avoid warning?)
  virtual ~eoRealVectorBounds(){}

  /** Default Ctor. I don't like it, as it leaves NULL pointers around
   */
  eoRealVectorBounds(unsigned _dim=0) : vector<eoRealBounds *>(_dim) {}

  /** Simple bounds = minimum and maximum (allowed)
  */
  eoRealVectorBounds(unsigned _dim, double _min, double _max) : 
    vector<eoRealBounds *>(_dim, new eoRealInterval(_min, _max))
  {
    if (_max-_min<=0)
      throw std::logic_error("Void range in eoRealVectorBounds");
  }

  /** Ctor: same bonds for everybody, given as an eoRealBounds
  */
  eoRealVectorBounds(unsigned _dim, eoRealBounds & _bounds) : 
    vector<eoRealBounds *>(_dim, &_bounds)
  {}
  
  /** Ctor: different bonds for different variables, vectors of double
   */
  eoRealVectorBounds(vector<double> _min, vector<double> _max) 
  {
    if (_max.size() != _min.size())
      throw std::logic_error("Dimensions don't match in eoRealVectorBounds");
    for (unsigned i=0; i<_min.size(); i++)
      {
	push_back( new eoRealInterval(_min[i], _max[i]));
      }
  }

  /** Ctor, particular case of dim-2
   */
  eoRealVectorBounds(eoRealBounds & _xbounds, eoRealBounds & _ybounds) : 
    vector<eoRealBounds *>(0)
  {
	push_back( &_xbounds);
	push_back( &_ybounds);
  }
  
  /** test: is i_th component bounded
   */
  virtual bool isBounded(unsigned _i) 
  { 
    return (*this)[_i]->isBounded();
  }
 
  /** test: bounded iff all are bounded
   */
  virtual bool isBounded(void) 
  {
    for (unsigned i=0; i<size(); i++)
      if (! (*this)[i]->isBounded())
	return false;
    return true;
  }

  /** Self-test: true iff i_th component has no bounds at all
   */
  virtual bool hasNoBoundAtAll(unsigned _i) 
  { 
    return (*this)[_i]->hasNoBoundAtAll();
  }
 
  /** Self-test: true iff all components have no bound at all
   */
  virtual bool hasNoBoundAtAll(void) 
  {
    for (unsigned i=0; i<size(); i++)
      if (! (*this)[i]->hasNoBoundAtAll())
	return false;
    return true;
  }

  virtual bool isMinBounded(unsigned _i) 
  { return (*this)[_i]->isMinBounded();} ;

  virtual bool isMaxBounded(unsigned _i) 
  { return (*this)[_i]->isMaxBounded();} ;

  /** Folds a real value back into the bounds - i_th component
   */
  virtual void foldsInBounds(unsigned _i, double & _r)
  {
    (*this)[_i]->foldsInBounds(_r);
  }

  /** Folds all variables of a vector of real values into the bounds
   */
  virtual void foldsInBounds(vector<double> & _v)
  {
   for (unsigned i=0; i<size(); i++)
     {
       (*this)[i]->foldsInBounds(_v[i]);
     }    
  }

  /** Truncates a real value to the bounds - i_th component
   */
  virtual void truncate(unsigned _i, double & _r)
  {
    (*this)[_i]->truncate(_r);
  }

  /** truncates all variables of a vector of real values to the bounds
   */
  virtual void truncate(vector<double> & _v)
  {
   for (unsigned i=0; i<size(); i++)
     {
       (*this)[i]->truncate(_v[i]);
     }    
  }

  /** test: is i_th component within the bounds?
   */
  virtual bool isInBounds(unsigned _i, double _r)
  { return (*this)[_i]->isInBounds(_r); }

  /** test: are ALL components within the bounds?
   */
  virtual bool isInBounds(vector<double> _v)
  {
    for (unsigned i=0; i<size(); i++)
      if (! isInBounds(i, _v[i]))
	return false;
    return true;
  }

  /** Accessors: will raise an exception if these do not exist
   */
  virtual double minimum(unsigned _i) {return (*this)[_i]->minimum();}
  virtual double maximum(unsigned _i) {return (*this)[_i]->maximum();}
  virtual double range(unsigned _i) {return (*this)[_i]->range();}

  /** Computes the average range
   *  An exception will be raised if one of the component is unbounded
   */
  virtual double averageRange() 
  {
    double r=0.0;
    for (unsigned i=0; i<size(); i++)
      r += range(i);
    return r/size();
  }

  /** Generates a random number in i_th range
   *  An exception will be raised if one of the component is unbounded
   */
  virtual double uniform(unsigned _i, eoRng & _rng = eo::rng)
  { 
    double r= (*this)[_i]->uniform();
    return r;
  }

  /** fills a vector with uniformly chosen variables in bounds
   *  An exception will be raised if one of the component is unbounded
   */
  void uniform(vector<double> & _v, eoRng & _rng = eo::rng)
  {
    _v.resize(size());
    for (unsigned i=0; i<size(); i++)
      {
      _v[i] = uniform(i, _rng);
      }
  }  
};

/** the dummy unbounded eoRealVectorBounds: usefull if you don't need bounds!
 * everything is inlined.
 * Warning: we do need this class, and not only a vector<eoRealNoBounds *>
 */
class eoRealVectorNoBounds: public eoRealVectorBounds
{ 
public:
  // virtual desctructor (to avoid warning?)
  virtual ~eoRealVectorNoBounds(){}

  /** 
   * Ctor: nothing to do, but beware of dimension: call base class ctor
   */
  eoRealVectorNoBounds(unsigned _dim=0) : eoRealVectorBounds(_dim)
  {
    // avoid NULL pointers, even though they shoudl (at the moment) never be used!
    if (_dim)
      for (unsigned i=0; i<_dim; i++)
	operator[](i)=&eoDummyRealNoBounds;
  }

  
  virtual bool isBounded(unsigned)  {return false;}
  virtual bool isBounded(void)   {return false;}

  virtual bool hasNoBoundAtAll(unsigned)  {return true;}
  virtual bool hasNoBoundAtAll(void)  {return true;}

  virtual bool isMinBounded(unsigned)   {return false;}
  virtual bool isMaxBounded(unsigned)   {return false;}

  virtual void foldsInBounds(unsigned, double &) {return;}
  virtual void foldsInBounds(vector<double> &) {return;}

  virtual void truncate(unsigned, double &) {return;}
  virtual void truncate(vector<double> &) {return;}

  virtual bool isInBounds(unsigned, double) {return true;}
  virtual bool isInBounds(vector<double>) {return true;}

  // accessors  
  virtual double minimum(unsigned)
  {
    throw logic_error("Trying to get minimum of eoRealVectorNoBounds");
  }
  virtual double maximum(unsigned)
  {
    throw logic_error("Trying to get maximum of eoRealVectorNoBounds");
  }
  virtual double range(unsigned)
  {
    throw logic_error("Trying to get range of eoRealVectorNoBounds");
  }

  virtual double averageRange() 
  {
    throw logic_error("Trying to get average range of eoRealVectorNoBounds");
  }

  // random generators
  virtual double uniform(unsigned, eoRng & _rng = eo::rng)
  {
    throw logic_error("No uniform distribution on eoRealVectorNoBounds");
  }

  // fills a vector with uniformly chosen variables in bounds
  void uniform(vector<double> &, eoRng & _rng = eo::rng)
  {
    throw logic_error("No uniform distribution on eoRealVectorNoBounds");
  }

};

// one object for all - see eoRealBounds.cpp
extern eoRealVectorNoBounds eoDummyVectorNoBounds;
#endif
