// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoRealVectorBounds.h
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

#ifndef _eoRealVectorBounds_h
#define _eoRealVectorBounds_h

#include <stdexcept>               // std::exceptions!
#include <utils/eoRNG.h>
#include <utils/eoRealBounds.h>

/**
Vector type for bounds (see eoRealBounds.h for scalar types)
------------
Class eoRealVectorBounds implements the std::vectorized version:
it is basically a std::vector of eoRealBounds * and forwards all request
to the elements of the std::vector.

This file also contains the global variables and eoDummyVectorNoBounds
that are used as defaults in ctors (i.e. when no
bounds are given, it is assumed unbounded values)

THe 2 main classes defined here are

eoRealBaseVectorBounds, base class that handles all useful functions
eoRealVectorBounds which derives from the preceding *and* eoPersistent
  and also has a mechanism for memory handling of the pointers
  it has to allocate

@ingroup Bounds
*/
class eoRealBaseVectorBounds : public std::vector<eoRealBounds *>
{
public:
  // virtual desctructor (to avoid warning?)
  virtual ~eoRealBaseVectorBounds(){}

  /** Default Ctor.
   */
  eoRealBaseVectorBounds() : std::vector<eoRealBounds *>(0) {}

  /** Ctor: same bounds for everybody, given as an eoRealBounds
  */
  eoRealBaseVectorBounds(unsigned _dim, eoRealBounds & _bounds) :
    std::vector<eoRealBounds *>(_dim, &_bounds)
  {}

  /** Ctor, particular case of dim-2
   */
  eoRealBaseVectorBounds(eoRealBounds & _xbounds, eoRealBounds & _ybounds) :
    std::vector<eoRealBounds *>(0)
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

  /** Folds all variables of a std::vector of real values into the bounds
   */
  virtual void foldsInBounds(std::vector<double> & _v)
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

  /** truncates all variables of a std::vector of real values to the bounds
   */
  virtual void truncate(std::vector<double> & _v)
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
  virtual bool isInBounds(std::vector<double> _v)
  {
    for (unsigned i=0; i<size(); i++)
      if (! isInBounds(i, _v[i]))
        return false;
    return true;
  }

  /** Accessors: will raise an std::exception if these do not exist
   */
  virtual double minimum(unsigned _i) {return (*this)[_i]->minimum();}
  virtual double maximum(unsigned _i) {return (*this)[_i]->maximum();}
  virtual double range(unsigned _i) {return (*this)[_i]->range();}

  /** Computes the average range
   *  An std::exception will be raised if one of the component is unbounded
   */
  virtual double averageRange()
  {
    double r=0.0;
    for (unsigned i=0; i<size(); i++)
      r += range(i);
    return r/size();
  }

  /** Generates a random number in i_th range
   *  An std::exception will be raised if one of the component is unbounded
   */
  virtual double uniform(unsigned _i, eoRng & _rng = eo::rng)
  {
    (void)_rng;

    double r= (*this)[_i]->uniform();
    return r;
  }

  /** fills a std::vector with uniformly chosen variables in bounds
   *  An std::exception will be raised if one of the component is unbounded
   */
  void uniform(std::vector<double> & _v, eoRng & _rng = eo::rng)
  {
    _v.resize(size());
    for (unsigned i=0; i<size(); i++)
      {
      _v[i] = uniform(i, _rng);
      }
  }

  /**
   * Write object. It's called printOn since it prints the object on a stream.
   * @param _os A std::ostream.
   */
  virtual void printOn(std::ostream& _os) const
  {
    for (unsigned i=0; i<size(); i++)
      {
        operator[](i)->printOn(_os);
        _os << ";";
      }
  }
};

////////////////////////////////////////////////////////////////////
/** Now a derived class, for parser reading
 * It holds some of the bounds (and destroy them when dying)

@ingroup Bounds
 */
class eoRealVectorBounds : public eoRealBaseVectorBounds, public eoPersistent
{
public:
  /** Default Ctor will call base class default ctor
   */
  eoRealVectorBounds():eoRealBaseVectorBounds() {}

  /** Ctor: same bounds for everybody, given as an eoRealBounds
  */
  eoRealVectorBounds(unsigned _dim, eoRealBounds & _bounds) :
    eoRealBaseVectorBounds(_dim, _bounds), factor(1,_dim), ownedBounds(0)
  {}

  /** Ctor, particular case of dim-2
   */
  eoRealVectorBounds(eoRealBounds & _xbounds, eoRealBounds & _ybounds) :
    eoRealBaseVectorBounds(_xbounds, _ybounds), factor(2,1), ownedBounds(0)
  {}

  /** Simple bounds = minimum and maximum (allowed)
  */
  eoRealVectorBounds(unsigned _dim, double _min, double _max) :
    eoRealBaseVectorBounds(), factor(1, _dim), ownedBounds(0)
  {
    if (_max-_min<=0)
      throw std::logic_error("Void range in eoRealVectorBounds");
    eoRealBounds *ptBounds = new eoRealInterval(_min, _max);
    // handle memory once
    ownedBounds.push_back(ptBounds);
    // same bound for everyone
    for (unsigned int i=0; i<_dim; i++)
      push_back(ptBounds);
  }

  /** Ctor: different bounds for different variables, std::vectors of double
   */
  eoRealVectorBounds(std::vector<double> _min, std::vector<double> _max) :
    factor(_min.size(), 1), ownedBounds(0)
  {
    if (_max.size() != _min.size())
      throw std::logic_error("Dimensions don't match in eoRealVectorBounds");
    // the bounds
    eoRealBounds *ptBounds;
    for (unsigned i=0; i<_min.size(); i++)
      {
        ptBounds = new eoRealInterval(_min[i], _max[i]);
        ownedBounds.push_back(ptBounds);
        push_back(ptBounds);
      }
  }

  /** Ctor from a std::string
   * and don't worry, the readFrom(std::string) starts by setting everything to 0!
  */
  eoRealVectorBounds(std::string _s) : eoRealBaseVectorBounds()
  {
    readFrom(_s);
  }

  /** Dtor: destroy all ownedBounds - BUG ???*/
  virtual ~eoRealVectorBounds()
  {
//     std::cout << "Dtor, avec size = " << ownedBounds.size() << std::endl;
//     for (unsigned i = 0; i < ownedBounds.size(); ++i)
//     {
//         delete ownedBounds[i];
//     }
}


  // methods from eoPersistent
  /**
   * Read object from a stream
   * only calls the readFrom(std::string) - for param reading
   * @param _is A std::istream.
   */
  virtual void readFrom(std::istream& _is) ;

  /**
   * Read object from a std::string
   * @param _s A std::istream.
   */
  virtual void readFrom(std::string _s) ;

  /** overload printOn method to save space */
  virtual void printOn(std::ostream& _os) const
  {
    if (factor[0]>1)
      _os << factor[0] ;
    operator[](0)->printOn(_os);

    // other bounds
    unsigned int index=factor[0];
    if (factor.size()>1)
      for (unsigned i=1; i<factor.size(); i++)
        {
          _os << ";";
          if (factor[i] > 1)
            _os << factor[i];
          operator[](index)->printOn(_os);
          index += factor[i];
        }
  }

  /** Eventually increases the size by duplicating last bound */
  void adjust_size(unsigned _dim);

  /** need to rewrite copy ctor and assignement operator
   *  because of ownedBounds */
  eoRealVectorBounds(const eoRealVectorBounds &);

private:// WARNING: there is no reason for both std::vector below
        //to be synchronized in any manner
  std::vector<unsigned int> factor;        // std::list of nb of "grouped" bounds
  std::vector<eoRealBounds *> ownedBounds;
// keep this one private
  eoRealVectorBounds& operator=(const eoRealVectorBounds&);
  };

//////////////////////////////////////////////////////////////
/** the dummy unbounded eoRealVectorBounds: usefull if you don't need bounds!
 * everything is inlined.
 * Warning: we do need this class, and not only a std::vector<eoRealNoBounds *>

@ingroup Bounds
 */
class eoRealVectorNoBounds: public eoRealVectorBounds
{
public:
  // virtual desctructor (to avoid warning?)
  virtual ~eoRealVectorNoBounds(){}

  /**
   * Ctor: nothing to do, but beware of dimension: call base class ctor
   */
  eoRealVectorNoBounds(unsigned _dim) :
    eoRealVectorBounds( (_dim?_dim:1), eoDummyRealNoBounds)
  {}


  virtual bool isBounded(unsigned)  {return false;}
  virtual bool isBounded(void)   {return false;}

  virtual bool hasNoBoundAtAll(unsigned)  {return true;}
  virtual bool hasNoBoundAtAll(void)  {return true;}

  virtual bool isMinBounded(unsigned)   {return false;}
  virtual bool isMaxBounded(unsigned)   {return false;}

  virtual void foldsInBounds(unsigned, double &) {return;}
  virtual void foldsInBounds(std::vector<double> &) {return;}

  virtual void truncate(unsigned, double &) {return;}
  virtual void truncate(std::vector<double> &) {return;}

  virtual bool isInBounds(unsigned, double) {return true;}
  virtual bool isInBounds(std::vector<double>) {return true;}

  // accessors
  virtual double minimum(unsigned)
  {
    throw std::logic_error("Trying to get minimum of eoRealVectorNoBounds");
  }
  virtual double maximum(unsigned)
  {
    throw std::logic_error("Trying to get maximum of eoRealVectorNoBounds");
  }
  virtual double range(unsigned)
  {
    throw std::logic_error("Trying to get range of eoRealVectorNoBounds");
  }

  virtual double averageRange()
  {
    throw std::logic_error("Trying to get average range of eoRealVectorNoBounds");
  }

  // random generators
  virtual double uniform(unsigned, eoRng & _rng = eo::rng)
  {
    (void)_rng;

    throw std::logic_error("No uniform distribution on eoRealVectorNoBounds");
  }

  // fills a std::vector with uniformly chosen variables in bounds
  void uniform(std::vector<double> &, eoRng & _rng = eo::rng)
  {
    (void)_rng;

    throw std::logic_error("No uniform distribution on eoRealVectorNoBounds");
  }

};



/** one object for all - see eoRealBounds.cpp
@ingroup Bounds
*/
extern eoRealVectorNoBounds eoDummyVectorNoBounds;
#endif
