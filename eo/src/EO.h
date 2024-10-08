// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// EO.h
// (c) GeNeura Team 1998
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
 */
//-----------------------------------------------------------------------------

#ifndef EO_H
#define EO_H

//-----------------------------------------------------------------------------

#include <stdexcept>       // std::runtime_error
#include "eoObject.h"      // eoObject
#include "eoPersistent.h"  // eoPersistent
#include "eoExceptions.h"

/**
    @defgroup Core Core components

    This are the base classes from which useful objects inherits.

    @{
 */

/** EO is the base class for objects with a fitness.

    Those evolvable objects are the subjects of
    evolution. EOs have only got a fitness, which at the same time needs to be
    only an object with the operation less than (<) defined. Fitness says how
    good is the object; evolution or change of these objects is left to the
    genetic operators.

    A fitness less than another means a worse fitness, in
    whatever the context; thus, fitness is always maximized; although it can
    be minimized with a proper definition of the < operator.

    A fitness can be invalid if undefined, trying to read an invalid fitness will raise an error.
    @ref Operators that effectively modify EO objects must invalidate them.

    The fitness object must have, besides an void ctor, a copy ctor.

    @example t-eo.cpp
*/
template<class F = double> class EO: public eoObject, public eoPersistent
{
public:
  typedef F Fitness;

  static constexpr const char* invalidTag = "INVALID";

  /** Default constructor.
  */
  EO(): repFitness(Fitness()), invalidFitness(true) { }

  /// Virtual dtor
  virtual ~EO() {};

  /// Return fitness value.
  // virtual const Fitness& fitness() const { // This would be impossible with MOEO.
  // virtual Fitness fitness() const { // Cannot do that either, MOEO changes the interface.
  Fitness fitness() const {
    if (invalid())
        throw eoInvalidFitnessError("Cannot retrieve unevaluated fitness");
    return repFitness;
  }

  /// Get fitness as reference, useful when fitness is set in a multi-stage way, e.g., MOFitness gets performance information, is subsequently ranked
  Fitness& fitnessReference() {
    if (invalid())
        throw eoInvalidFitnessError("Cannot retrieve unevaluated fitness reference");
    return repFitness;
  }

  // Set fitness as invalid.
  virtual void invalidate() { invalidFitness = true; repFitness = Fitness(); }

  /** Set fitness. At the same time, validates it.
   *  @param _fitness New fitness value.
   */
  void fitness(const Fitness& _fitness)
  {
    repFitness = _fitness;
    invalidFitness = false;
  }

  /** Return true If fitness value is invalid, false otherwise.
   *  @return true If fitness is invalid.
   */
  bool invalid() const { return invalidFitness; }

  /** Returns true if
      @return true if the fitness is higher
  */
  bool operator<(const EO& _eo2) const { return fitness() < _eo2.fitness(); }
  bool operator>(const EO& _eo2) const { return !(fitness() <= _eo2.fitness()); }

  /// Methods inherited from eoObject
  //@{

  /** Return the class id.
   *  @return the class name as a std::string
   */
  virtual std::string className() const { return "EO"; }

  /**
   * Read object.\\
   * Calls base class, just in case that one had something to do.
   * The read and print methods should be compatible and have the same format.
   * In principle, format is "plain": they just print a number
   * @param _is a std::istream.
   */
  virtual void readFrom(std::istream& _is)
  {
        std::string fitness_str;
        _is >> fitness_str;

        if (fitness_str == invalidTag)
        {
                invalidFitness = true;
        }
        else
        {
                invalidFitness = false;
                std::istringstream iss(fitness_str);
                iss >> repFitness;
        }
  }

  /**
   * Write object. Called printOn since it prints the object _on_ a stream.
   * @param _os A std::ostream.
   */
  virtual void printOn(std::ostream& _os) const
  {
    if (invalid())
    {
        _os << invalidTag << ' ';
    }
    else
    {
        _os << repFitness << ' ';
    }

  }

  //@}

private:
  Fitness repFitness;   // value of fitness for this chromosome
  bool invalidFitness;  // true if the value of fitness is invalid
};


#endif

/** @} */
