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

/** @addtogroup Evaluation
 * @{
 */

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
/** @example t-eoFitnessAssembled.cpp
 */

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
template <class ScalarType, class Compare, class FitnessTraits >
class eoScalarFitnessAssembled : public std::vector<ScalarType> {

public:

    using std::vector< ScalarType >::empty;
    using std::vector< ScalarType >::front;
    using std::vector< ScalarType >::size;


  typedef typename std::vector<ScalarType> baseVector;
  typedef typename baseVector::size_type size_type;

  // Basic constructors and assignments
  eoScalarFitnessAssembled()
    : baseVector( FitnessTraits::size() ),
      feasible(true), failed(false), msg("")
  {}

  eoScalarFitnessAssembled( size_type _n,
                            const ScalarType& _val,
                            const std::string& _descr="Unnamed variable" )
    : baseVector(_n, _val),
      feasible(true), failed(false), msg("")
  {
    if ( _n > FitnessTraits::size() )
    FitnessTraits::resize(_n, _descr);
  }

  eoScalarFitnessAssembled( const eoScalarFitnessAssembled& other)
    : baseVector( other ),
      feasible(other.feasible),
      failed(other.failed),
      msg(other.msg)
  {}

  eoScalarFitnessAssembled& operator=( const eoScalarFitnessAssembled& other) {
    baseVector::operator=( other );
    feasible = other.feasible;
    failed = other.failed;
    msg = other.msg;
    return *this;
  }

  // Constructors and assignments to work with scalar type
  eoScalarFitnessAssembled( const ScalarType& v )
    : baseVector( 1, v ),
      feasible(true), failed(false), msg("")
  {}

  eoScalarFitnessAssembled& operator=( const ScalarType& v ) {

      if( empty() )
          push_back( v );
      else
          front() = v;
      return *this;
  }

  //! Overload push_back()
  void push_back(const ScalarType& _val ){
    baseVector::push_back( _val );
    if ( size() > FitnessTraits::size() )
      FitnessTraits::setDescription( size()-1, "Unnamed variable");
  }

  //! Overload push_back()
  void push_back(const ScalarType& _val, const std::string& _descr ){
    baseVector::push_back( _val );
    FitnessTraits::setDescription( size()-1, _descr );
  }

  //! Overload resize()
  void resize( size_type _n, const ScalarType& _val = ScalarType(), const std::string& _descr = "Unnamed variable" ){
    baseVector::resize(_n, _val);
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

  //! Feasibility boolean
  /**
   * Can be specified anywhere in fitness evaluation
   * as an indicator if the individual is in some feasible range.
   */
  bool feasible;

  //! Failed boolean
  /**
   * Can be specified anywhere in fitness evaluation
   * as an indicator if the evaluation of the individual failed
   */
  bool failed;

  //! Message
  /**
   * Can be specified anywhere in fitness evaluation.
   * Typically used to store some sort of error messages, if evaluation of individual failed.
   */
  std::string msg;


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
      os << FitnessTraits::getDescription(i) << " = " << this->operator[](i) << " ";
  }

  //! Comparison, using less by default
  bool operator<(const eoScalarFitnessAssembled& other) const{
    if ( empty() || other.empty() )
      return false;
    else
      return Compare()( front() , other.front() );
  }

  //! Comparison with ScalarTypes. Explicit definition needed to compile with VS 8.0
  bool operator<(ScalarType x) const{
        eoScalarFitnessAssembled ScalarFitness(x);
        return this->operator<(ScalarFitness);
  }

  // implementation of the other operators
  bool operator>( const eoScalarFitnessAssembled<ScalarType, Compare, FitnessTraits>& y ) const  { return y < *this; }

  // implementation of the other operators
  bool operator<=( const eoScalarFitnessAssembled<ScalarType, Compare, FitnessTraits>& y ) const { return !(*this > y); }

  // implementation of the other operators
  bool operator>=(const eoScalarFitnessAssembled<ScalarType, Compare, FitnessTraits>& y ) const { return !(*this < y); }

};
/**
 * @example t-eoFitnessAssembledEA.cpp
*/


/**
Typedefs for fitness comparison, Maximizing Fitness compares with less,
and minimizing fitness compares with greater. This because we want ordinary
fitness values (doubles) to be equivalent with Maximizing Fitness, and
comparing with less is the default behaviour.
*/
typedef eoScalarFitnessAssembled<double, std::less<double>, eoScalarFitnessAssembledTraits >    eoAssembledMaximizingFitness;
typedef eoScalarFitnessAssembled<double, std::greater<double>, eoScalarFitnessAssembledTraits > eoAssembledMinimizingFitness;

template <class F, class Cmp, class FitnessTraits>
std::ostream& operator<<(std::ostream& os, const eoScalarFitnessAssembled<F, Cmp, FitnessTraits>& f)
{
  for (unsigned i=0; i < f.size(); ++i)
    os << f[i] << " ";

  os << f.feasible << " ";
  os << f.failed << " ";

  return os;
}

template <class F, class Cmp, class FitnessTraits>
std::istream& operator>>(std::istream& is, eoScalarFitnessAssembled<F, Cmp, FitnessTraits>& f)
{
  for (unsigned i=0; i < f.size(); ++i){
    F value;
    is >> value;
    f[i] = value;
  }

  is >> f.feasible;
  is >> f.failed;

  return is;
}

/** @} */
#endif
