 /* -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*- */

//-----------------------------------------------------------------------------
// eoScalarFitnessAssembled.h
// Marc Wintermantel & Oliver Koenig
// IMES-ST@ETHZ.CH
// March 2003

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
             Marc.Schoenauer@inria.fr
             mak@dhi.dk
*/
//-----------------------------------------------------------------------------

#ifndef eoScalarFitnessAssembled_h
#define eoScalarFitnessAssembled_h

#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>

//! Defines properties of eoScalarFitnessAssembled.
/*! Properties that are hold in this traits class:
    - std::vector<std::string> to hold descriptions of the different fitness terms
*/
class eoScalarFitnessAssembledTraits{

public:
  
  typedef std::vector<std::string>::size_type size_type;

  static void setDescription( size_type _idx, std::string _descr ) { 
    if ( _idx < TermDescriptions.size() )
      TermDescriptions[_idx] = _descr; 
    else{
      TermDescriptions.resize(_idx, "Unnamed variable" );
      TermDescriptions[_idx] = _descr;
    }
  }

  static std::string getDescription( size_type _idx) { 
    if ( _idx < TermDescriptions.size() )
      return TermDescriptions[_idx ]; 
    else
      return "Unnamed Variable";
  }
  
  static void resize( size_type _n, const std::string& _descr) { 
    TermDescriptions.resize(_n, _descr); 
  }
  
  static size_type size() { return TermDescriptions.size(); }

  static std::vector<std::string> getDescriptionVector() { return TermDescriptions; }
  
private:
  static std::vector<std::string> TermDescriptions;
};

//! Implements fitness as std::vector, storing all values that might occur during fitness assembly
/*! Properties:
    - Wraps a scalar fitness values such as a double or int, with the option of
      maximizing (using less<ScalarType>) or minimizing (using greater<ScalarType>).
    - Stores all kinda different values met during fitness assembly, to be defined in eoEvalFunc.
    - It overrides operator<() to use the Compare template argument.
    - Suitable constructors and assignments and casts are defined to work 
      with this quantity as if it were a ScalarType.
    - Global fitness value is stored as first element in the vector
*/
template <class ScalarType, class Compare, class FitnessTraits = eoScalarFitnessAssembledTraits >
class eoScalarFitnessAssembled : public std::vector<ScalarType> {

public:
  
  typedef typename std::vector<ScalarType>::size_type size_type;

  // Basic constructors and assignments
  eoScalarFitnessAssembled() 
    : std::vector<ScalarType>( FitnessTraits::size() ) {}

  eoScalarFitnessAssembled( size_type _n, 
			    const ScalarType& _val,
			    const std::string& _descr="Unnamed variable" )
    : std::vector<ScalarType>(_n, _val) 
  { 
    if ( _n > FitnessTraits::size() )
    FitnessTraits::resize(_n, _descr);
  }
  
  eoScalarFitnessAssembled( const eoScalarFitnessAssembled& other) : std::vector<ScalarType>( other ) {}

  eoScalarFitnessAssembled& operator=( const eoScalarFitnessAssembled& other) {
#ifdef _MSC_VER
    typedef std::vector<ScalarType> myvector;
    myvector::operator=( other );
#else
    std::vector<ScalarType>::operator=( other );
#endif
    return *this;
  }
  
  // Constructors and assignments to work with scalar type
  eoScalarFitnessAssembled( const ScalarType& v ) : std::vector<ScalarType>( 1, v ) {}
  eoScalarFitnessAssembled& operator=( const ScalarType& v ) {
    
    if ( empty() )
      push_back( v );
    else
      front() = v;
    
    return *this;
  }
  
  //! Overload push_back()
  void push_back(const ScalarType& _val ){
#ifdef _MSC_VER
    typedef std::vector<ScalarType> myvector;
    myvector::push_back( _val );
#else
    std::vector<ScalarType>::push_back( _val );
#endif
    if ( size() > FitnessTraits::size() ) 
      FitnessTraits::setDescription( size()-1, "Unnamed variable");
  }

  //! Overload push_back()
  void push_back(const ScalarType& _val, const std::string& _descr ){
#ifdef _MSC_VER
    typedef std::vector<ScalarType> myvector;
    myvector::push_back( _val );
#else
    std::vector<ScalarType>::push_back( _val );
#endif
    FitnessTraits::setDescription( size()-1, _descr );
  }

  //! Overload resize()
  void resize( size_type _n, const ScalarType& _val = ScalarType(), const std::string& _descr = "Unnamed variable" ){
    std::vector<ScalarType>::resize(_n, _val);
    FitnessTraits::resize(_n, _descr);
  }
  
  //! Set description
  void setDescription( size_type _idx, std::string _descr ) { 
    FitnessTraits::setDescription( _idx, _descr );
  }
  
  //! Get description
  std::string getDescription( size_type _idx ){ return FitnessTraits::getDescription( _idx ); }

  //! Get vector with descriptions
  std::vector<std::string> getDescriptionVector() { return FitnessTraits::getDescriptionVector(); }

  // Scalar type access
  operator ScalarType(void) const { 
    if ( empty() )
      return 0.0;
    else
      return front(); 
  }

  //! Print term values and descriptions
  void printAll(std::ostream& os) const {
    for (size_type i=0; i < size(); ++i )
      os << FitnessTraits::getDescription(i) << " = " << operator[](i) << " ";
  }

  // Comparison, using less by default
  bool operator<(const eoScalarFitnessAssembled& other) const{ 
    if ( empty() || other.empty() )
      return false;
    else
      return Compare()( front() , other.front() ); 
  }

  // implementation of the other operators
  bool operator>( const eoScalarFitnessAssembled<ScalarType, Compare>& y ) const  { return y < *this; }

  // implementation of the other operators
  bool operator<=( const eoScalarFitnessAssembled<ScalarType, Compare>& y ) const { return !(*this > y); }

  // implementation of the other operators
  bool operator>=(const eoScalarFitnessAssembled<ScalarType, Compare>& y ) const { return !(*this < y); }

};

/**
Typedefs for fitness comparison, Maximizing Fitness compares with less,
and minimizing fitness compares with greater. This because we want ordinary
fitness values (doubles) to be equivalent with Maximizing Fitness, and
comparing with less is the default behaviour.
*/
typedef eoScalarFitnessAssembled<double, std::less<double> >    eoAssembledMaximizingFitness;
typedef eoScalarFitnessAssembled<double, std::greater<double> > eoAssembledMinimizingFitness;

template <class F, class Cmp>
std::ostream& operator<<(std::ostream& os, const eoScalarFitnessAssembled<F, Cmp>& f)
{
  for (unsigned i=0; i < f.size(); ++i)
    os << f[i] << " ";
    return os;
}

template <class F, class Cmp>
std::istream& operator>>(std::istream& is, eoScalarFitnessAssembled<F, Cmp>& f)
{
  for (unsigned i=0; i < f.size(); ++i){
    F value;
    is >> value;
    f[i] = value;
  }
   
  return is;
}

#endif



