// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoRandomSelect.h
// (c) GeNeura Team, 1998
//-----------------------------------------------------------------------------

#ifndef EORANDOMSELECT_H
#define EORANDOMSELECT_H

//-----------------------------------------------------------------------------

#include <algorithm>
#include <numeric>    // for accumulate
#include <functional>

#include <eoPopOps.h>
#include <eoUniform.h>

//-----------------------------------------------------------------------------

/** 
 * eoRandomSelect: an selection operator, which selects randomly a percentage
 of the initial population.
 */
template<class EOT> class eoRandomSelect: public eoSelect<EOT>
{
 public:
  ///
  eoRandomSelect(const float& _percent = 0.4): eoSelect<EOT>(), rate(_percent) {};
  
  ///
  virtual ~eoRandomSelect() {};
  
  /// Takes a percentage of the population randomly, and transfers it to siblings
  virtual void operator() ( const eoPop<EOT>& _parents, eoPop<EOT>& _siblings ) const {
    // generates random numbers
    eoUniform<unsigned> uniform(0, _parents.size()-1);
    unsigned num_chroms = (unsigned)(rate * _parents.size());

    // selection of chromosomes
    do {
      _siblings.push_back(_parents[uniform()]);
    } while (_siblings.size() < num_chroms);
  }

    /// @name Methods from eoObject
  //@{
  /**
   * Read object. Reads the percentage
   * Should call base class, just in case.
   * @param _s A istream.
   */
  virtual void readFrom(istream& _s) {
	_s >> rate;
  }

  /** Print itself: inherited from eoObject implementation. Declared virtual so that 
      it can be reimplemented anywhere. Instance from base classes are processed in
	  base classes, so you don´t have to worry about, for instance, fitness.
  @param _s the ostream in which things are written*/
  virtual void printOn( ostream& _s ) const{
	_s << rate;
  }

  /** Inherited from eoObject 
      @see eoObject
  */
  string className() const {return "eoRandomSelect";};

  //@}


 private:
  float rate;
};

//-----------------------------------------------------------------------------

#endif EOGSRANDOMSELECT_H
