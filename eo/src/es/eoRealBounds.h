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
- bool isInBounds(double &) that simply says whether or not the argument 
  is in the bounds

So mutation can choose whetehr they want to iterate trying until 
they fall in bounds, or only try once and "repair" by using 
the foldsInBounds method

There is also a uniform() method that generates a uniform value 
(if possible, i.e. if bounded) in the interval.

Derived class are 
 eoRealInterval, that holds a minimum and maximum value
 eoRealNoBounds, that implements the "unbounded bounds"

TODO: the eoRealMinBound and eoRealMaxBound that implement 
      the half-bounded intervals.

Vector type: 
------------
Class eoRealVectorBounds implements the vectorized version: 
it is basically a vector of eoRealBounds * and forwards all request
to the elements of the vector.

*/
class eoRealBounds
{ 
public:
  virtual bool isBounded(void) = 0;
  virtual bool isMinBounded(void) = 0;
  virtual bool isMaxBounded(void) = 0;
  virtual void foldsInBounds(double &) = 0;
  virtual bool isInBounds(double) = 0;

  // accessors  
  virtual double minimum() = 0;
  virtual double maximum() = 0;
  virtual double range() = 0;

  // random generators
  virtual double uniform(eoRng & _rng = eo::rng) = 0;
};

class eoRealNoBounds : public eoRealBounds
{
public:
  virtual ~eoRealNoBounds(){}

  virtual bool isBounded(void) {return false;}
  virtual bool isMinBounded(void) {return false;}
  virtual bool isMaxBounded(void) {return false;}
  virtual void foldsInBounds(double &) {return;}
  virtual bool isInBounds(double) {return true;}

  // accessors  
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

  // random generators
  virtual double uniform(eoRng & _rng = eo::rng)
  {
    throw logic_error("Trying to generate uniform values in unbounded eoRealBounds");
  }
};

/* fully bounded == interval */
class eoRealInterval : public eoRealBounds
{
public :
  
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
  virtual bool isMinBounded(void) {return true;}
  virtual bool isMaxBounded(void) {return true;}

  double uniform(eoRng & _rng = eo::rng)
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
  void foldsInBounds(double &  _r)
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

private :
  double repMinimum;
  double repMaximum;
  double repRange;			   // to minimize operations ???
};


// now the vectorized version

class eoRealVectorBounds : public vector<eoRealBounds *>
{ 
public:
  // virtual desctructor (to avoid warining?)
  virtual ~eoRealVectorBounds(){}

  // Default Ctor
  eoRealVectorBounds() : 
    vector<eoRealBounds *>(0) {}

  /** 
      Simple bounds = minimum and maximum (allowed)
  */
  // Ctor: same bonds for everybody, explicit
  eoRealVectorBounds(unsigned _dim, double _min, double _max) : 
    vector<eoRealBounds *>(_dim, new eoRealInterval(_min, _max))
  {
    if (_max-_min<=0)
      throw std::logic_error("Void range in eoRealVectorBounds");
  }

  // Ctor: same bonds for everybody, given as a eoRealBounds
  eoRealVectorBounds(unsigned _dim, eoRealBounds & _bounds) : 
    vector<eoRealBounds *>(_dim, &_bounds)
  {}
  
  // Ctor: different bonds for different variables, vectors of double
  eoRealVectorBounds(vector<double> _min, vector<double> _max) 
  {
    if (_max.size() != _min.size())
      throw std::logic_error("Dimensions don't match in eoRealVectorBounds");
    for (unsigned i=0; i<_min.size(); i++)
      {
	push_back( new eoRealInterval(_min[i], _max[i]));
      }
  }

  // Ctor, particular case of dim-2
  eoRealVectorBounds(eoRealBounds & _xbounds, eoRealBounds & _ybounds) : 
    vector<eoRealBounds *>(0)
  {
	push_back( &_xbounds);
	push_back( &_ybounds);
  }
  
  virtual bool isBounded(unsigned _i) 
  { 
    return (*this)[_i]->isBounded();
  }
 
  // bounded iff all are bounded
  virtual bool isBounded(void) 
  {
    for (unsigned i=0; i<size(); i++)
      if (! (*this)[i]->isBounded())
	return false;
    return true;
  }

  // these do not make any sense as vectors!
  virtual bool isMinBounded(unsigned _i) 
  { return (*this)[_i]->isMinBounded();} ;

  virtual bool isMaxBounded(unsigned _i) 
  { return (*this)[_i]->isMaxBounded();} ;

  virtual void foldsInBounds(unsigned _i, double & _r)
  {
    (*this)[_i]->foldsInBounds(_r);
  }

  virtual void foldsInBounds(vector<double> & _v)
  {
   for (unsigned i=0; i<size(); i++)
     {
       foldsInBounds(i, _v[i]);
     }    
  }

  virtual bool isInBounds(unsigned _i, double _r)
  { return (*this)[_i]->isInBounds(_r); }

  // isInBounds iff all are in bouds
  virtual bool isInBounds(vector<double> _v)
  {
    for (unsigned i=0; i<size(); i++)
      if (! isInBounds(i, _v[i]))
	return false;
    return true;
  }

  // accessors  
  virtual double minimum(unsigned _i) {return (*this)[_i]->minimum();}
  virtual double maximum(unsigned _i) {return (*this)[_i]->maximum();}
  virtual double range(unsigned _i) {return (*this)[_i]->range();}

  virtual double averageRange() 
  {
    double r=0.0;
    for (unsigned i=0; i<size(); i++)
      r += range(i);
    return r/size();
  }

  // random generators
  virtual double uniform(unsigned _i, eoRng & _rng = eo::rng)
  { 
    double r= (*this)[_i]->uniform();
    return r;
  }

  // fills a vector with uniformly chosen variables in bounds
  void uniform(vector<double> & _v, eoRng & _rng = eo::rng)
  {
    _v.resize(size());
    for (unsigned i=0; i<size(); i++)
      {
      _v[i] = uniform(i, _rng);
      }
  }  
};

// the dummy unbounded eoRealVectorBounds:

class eoRealVectorNoBounds: public eoRealVectorBounds
{ 
public:
  // virtual desctructor (to avoid warining?)
  virtual ~eoRealVectorNoBounds(){}

  /** 
      Simple bounds = minimum and maximum (allowed)
  */
  // Ctor: nothing to do!
  eoRealVectorNoBounds(unsigned _dim=0) {}

  
  virtual bool isBounded(unsigned)  {return false;}
  virtual bool isBounded(void)   {return false;}
  virtual bool isMinBounded(unsigned)   {return false;}
  virtual bool isMaxBounded(unsigned)   {return false;}

  virtual void foldsInBounds(unsigned, double &) {return;}
  virtual void foldsInBounds(vector<double> &) {return;}

  virtual bool isInBounds(unsigned, double) {return true;}
  virtual bool isInBounds(vector<double>) {return true;}

  // accessors  
  virtual double minimum(unsigned)
  {
    throw logic_error("Trying to get minimum of unbounded eoRealBounds");
  }
  virtual double maximum(unsigned)
  {
    throw logic_error("Trying to get maximum of unbounded eoRealBounds");
  }
  virtual double range(unsigned)
  {
    throw logic_error("Trying to get range of unbounded eoRealBounds");
  }

  virtual double averageRange() 
  {
    throw logic_error("Trying to get average range of unbounded eoRealBounds");
  }

  // random generators
  virtual double uniform(unsigned, eoRng & _rng = eo::rng)
  {
    throw logic_error("No uniform distribution on unbounded eoRealBounds");
  }

  // fills a vector with uniformly chosen variables in bounds
  void uniform(vector<double> &, eoRng & _rng = eo::rng)
  {
    throw logic_error("No uniform distribution on unbounded eoRealBounds");
  }

};

// one object for all
eoRealNoBounds eoDummyRealNoBounds;
eoRealVectorNoBounds eoDummyVectorNoBounds;
#endif
