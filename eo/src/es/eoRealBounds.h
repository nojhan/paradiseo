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

  Various classes for the initialization and mutation of real valued vectors.

  Supports simple mutations and various more adaptable mutations, including
  correlated mutations.

*/


/**
\class eoRealBounds eoRealBounds.h es/eoRealBounds.h
\ingroup EvolutionStrategies

    Defines the minima and maxima for real variables

*/
class eoBaseRealBounds : public eoUF<double, bool>
{ };

class eoRealBounds : public eoBaseRealBounds
{
public :
  
  /** 
      Simple bounds = minimum and maximum (allowed)
  */
  eoRealBounds(double _min=0, double _max=1) : 
    repMinimum(_min), repMaximum(_max), repRange(_max-_min) 
  {
    if (repRange<=0)
      throw std::logic_error("Void range in eoRealBounds");
  }
  
  double Minimum() { return repMinimum; }
  double Maximum() { return repMaximum; }
  double Range()   { return repRange; }
  // for backward compatibility
  double minimum() { return repMinimum; }
  double maximum() { return repMaximum; }
  double range()   { return repRange; }

  double uniform(eoRng & _rng = eo::rng)
  {
    return repMinimum + _rng.uniform(repRange);
  }  

  // says if a given double is within the bounds
  bool operator()(double _r)
  {
    if (_r < repMinimum)
      return false;
    if (_r > repMaximum)
      return false;
    return true;
  }

private :
  double repMinimum;
  double repMaximum;
  double repRange;			   // to minimize operations ???
};


// now the vectorized version

class eoRealVectorBounds 
{
public :
  
  /** 
      Simple bounds = minimum and maximum (allowed)
  */
  // Ctor: same bonds for everybody, explicit
  eoRealVectorBounds(unsigned _dim, double _min=0, double _max=1) : 
    vecMinimum(_dim, _min), vecMaximum(_dim, _max), vecRange(_dim, _max-_min) 
  {
    if (_max-_min<=0)
      throw std::logic_error("Void range in eoRealVectorBounds");
  }

  // Ctor: same bonds for everybody, given as a eoRealBounds
  eoRealVectorBounds(unsigned _dim, eoRealBounds & _bounds) : 
    vecMinimum(_dim, _bounds.Minimum()), 
    vecMaximum(_dim, _bounds.Maximum()), 
    vecRange(_dim, _bounds.Range()) 
  {}
  
  // Ctor: different bonds for different variables, vectors of double
  eoRealVectorBounds(vector<double> _min, vector<double> _max) : 
    vecMinimum(_min), vecMaximum(_max), vecRange(_min.size()) 
  {
    if (_max.size() != _min.size())
      throw std::logic_error("Dimensions don't match in eoRealVectorBounds");
    for (unsigned i=0; i<_min.size(); i++)
      {
	vecRange[i]=_max[i]-_min[i];
	if (vecRange[i]<=0)
	  throw std::logic_error("Void range in eoRealVectorBounds");
      }
  }

  // Ctor, particular case of dim-2
  eoRealVectorBounds(eoRealBounds & _xbounds, eoRealBounds & _ybounds) : 
    vecMinimum(2), vecMaximum(2), vecRange(2) 
  {
    vecMinimum[0] = _xbounds.Minimum();
    vecMaximum[0] = _xbounds.Maximum();
    vecRange[0] = _xbounds.Range();
    vecMinimum[1] = _ybounds.Minimum();
    vecMaximum[1] = _ybounds.Maximum();
    vecRange[1] = _ybounds.Range();    
  }
  
  // not a ctor, but usefull to initialize, too
  // is it safe to call it push_back? Maybe not, but it's meaningful!
  void push_back(double _min=0, double _max=1)
  {
    vecMinimum.push_back(_min);
    vecMaximum.push_back(_max);
    if (_max-_min <= 0)
      throw std::logic_error("Void range in eoRealVectorBounds::add");
    vecRange.push_back(_max-_min);
  }

  void push_back(eoRealBounds & _bounds)
  {
    vecMinimum.push_back(_bounds.Minimum());
    vecMaximum.push_back(_bounds.Maximum());
    vecRange.push_back(_bounds.Range());
  }

  // accessors - following rule that says that method start are capitalized
  double Minimum(unsigned _i) { return vecMinimum[_i]; }
  double Maximum(unsigned _i) { return vecMaximum[_i]; }
  double Range(unsigned _i)   { return vecRange[_i]; }

  // accessors - for backward compatibility
  double minimum(unsigned _i) { return vecMinimum[_i]; }
  double maximum(unsigned _i) { return vecMaximum[_i]; }
  double range(unsigned _i)   { return vecRange[_i]; }

  // handy: get the size
  unsigned int size() { return vecMinimum.size();}

  // returns a value uniformly chosen in bounds for a given variable
  double uniform(unsigned _i, eoRng & _rng = eo::rng)
  {
    return vecMinimum[_i] + _rng.uniform(vecRange[_i]);
  }  

  // returns a vector of uniformly chosen variables in bounds
  vector<double> uniform(eoRng & _rng = eo::rng)
  {
    vector<double> v(vecMinimum.size());
    for (unsigned i=0; i<vecMinimum.size(); i++)
      v[i] = vecMinimum[i] + _rng.uniform(vecRange[i]);

    return v;
  }  

  // fills a vector with uniformly chosen variables in bounds
  void uniform(vector<double> & _v, eoRng & _rng = eo::rng)
  {
    _v.resize(vecMinimum.size());
    for (unsigned i=0; i<vecMinimum.size(); i++)
      _v[i] = vecMinimum[i] + _rng.uniform(vecRange[i]);
  }  

  // says if a given double is within the bounds
  bool operator()(unsigned _i, double _r)
  {
    if (_r < vecMinimum[_i])
      return false;
    if (_r > vecMaximum[_i])
      return false;
    return true;
  }

  // check the bounds for a vector: true only if ALL ar ein bounds
  bool operator()(vector<double> & _v)
  {
    for (unsigned i=0; i<_v.size(); i++)
      if (! operator()(i, _v[i]) ) // out of bound
	return false;
    return true;
  }
private :
  vector<double> vecMinimum;
  vector<double> vecMaximum;
  vector<double> vecRange;			   // to minimize operations ???
};

#endif
