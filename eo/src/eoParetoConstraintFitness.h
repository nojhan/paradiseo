// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoParetoConstraintFitness.h
// (c) Maarten Keijzer and Marc Schoenauer, 2001
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

    Contact: mkeijzer@cs.vu.nl
             Marc.Schoenauer@inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef _eoParetoConstraintFitness_h
#define _eoParetoConstraintFitness_h

#include <math.h>
#include <vector>
#include <stdexcept>
#include <iostream>

#include <eoParetoFitness.h>

/**
  eoParetoOneConstraintFitness class: 
            std::vector of doubles + constraint value

  Comparison (dominance) is done 
          on pareto dominance for 2 feasible individuals, 
	  one feasible individual always wins over an infeasible
	  on constraint violations for  2 infeasible individuals

  The template argument FitnessTraits defaults to
  eoParetoFitnessTraits, which can be replaces at will by any other
  class that implements the static functions defined therein. 

  Note that the domninance defines a partial order, so that
    !dominates(a,b) && !domaintes(b,a) does not neccessarily imply that (a==b)
  The other way around does hold.

  However, be careful that the comparison operators do define a total order
  based on considering first objective, then in case of tie, second objective, etc

  NOTE: in a hurry, I did not want to make it derive from eoParetoFitness  
  (used cut-and-paste instead!) : I know it might be a good idea, but I'm 
  not sure I see why at the moment (any hint someone?) 
*/
template <class FitnessTraits = eoParetoFitnessTraits>
class eoParetoOneConstraintFitness : public std::vector<double>
{
private: 
  // this class implements only 1 inequality constraint 
  //               (must ponder a bit for generality without huge overload)
  double constraintValue;	   // inequality cstr - must be negative

public :
  typedef FitnessTraits fitness_traits;

  eoParetoOneConstraintFitness(void) : std::vector<double>(FitnessTraits::nObjectives(),0.0)  {}

  // Ctr from a std::vector<double> (size nObjectives+1)
  eoParetoOneConstraintFitness(std::vector<double> & _v) : 
    std::vector<double>(_v) 
  {
#ifndef NDEBUG
    if (_v.size() != fitness_traits::nObjectives()+1)
      throw std::runtime_error("Size error in Ctor of eoParetoOneConstraintFitness from std::vector");
#endif
    constraintValue = _v[fitness_traits::nObjectives()];
    resize(fitness_traits::nObjectives());
  }
  
  // Ctr from a std::vector<double> and a value
  eoParetoOneConstraintFitness(std::vector<double> & _v, double _c) : 
    std::vector<double>(_v), constraintValue(_c) 
  {
#ifndef NDEBUG
    if (_v.size() != fitness_traits::nObjectives())
      throw std::runtime_error("Size error in Ctor of eoParetoOneConstraintFitness from std::vector and value");
#endif
  }
  

  /** access to the traits characteristics (so you don't have to write 
   * a lot of typedef's around
   */
  static void setUp(unsigned _n, std::vector<bool> & _b) {FitnessTraits::setUp(_n, _b);}
  static bool maximizing(unsigned _i) { return FitnessTraits::maximizing(_i);}

  bool feasible() const { return constraintValue<=0;} 
  double violation() const { return (feasible()?0.0:constraintValue);}
  double ConstraintValue() const {return constraintValue;}
  void ConstraintValue(double _c) {constraintValue=_c;}

  /// Partial order based on Pareto-dominance
  //bool operator<(const eoParetoFitness<FitnessTraits>& _other) const
  bool dominates(const eoParetoOneConstraintFitness<FitnessTraits>& _other) const
  {
    bool dom = false;

    double tol = FitnessTraits::tol();
    const std::vector<double>& performance = *this;
    const std::vector<double>& otherperformance = _other;

    if (feasible() && _other.feasible())
    // here both are feasible: do the "standard" domination
      for (unsigned i = 0; i < FitnessTraits::nObjectives(); ++i)
	{
	  bool maxim = FitnessTraits::maximizing(i);
	  double aval = maxim? performance[i] : -performance[i];
	  double bval = maxim? otherperformance[i] : -otherperformance[i];
	  
	  if (fabs(aval - bval) > tol)
	    {
	      if (aval < bval)
		{
		  return false; // cannot dominate
		}
	      // else aval < bval
	      dom = true; // for the moment: goto next objective
	    }
	  //else they're equal in this objective, goto next
	}
    else
      {			   // one at least is not feasible
	if (feasible())
	  return true;		   // feasible wins
	else if (_other.feasible())
	  return false;		   // feasible wins
	return (violation()<_other.violation()); // smallest violation wins
      }

    return dom;
  }

  /// compare *not* on dominance, but on the first, then the second, etc
  bool operator<(const eoParetoOneConstraintFitness<FitnessTraits>& _other) const
  {
    double tol = FitnessTraits::tol();
    const std::vector<double>& performance = *this;
    const std::vector<double>& otherperformance = _other;
    for (unsigned i = 0; i < FitnessTraits::nObjectives(); ++i)
    {
      bool maxim = FitnessTraits::maximizing(i);
      double aval = maxim? performance[i] : -performance[i];
      double bval = maxim? otherperformance[i] : -otherperformance[i];

      if (fabs(aval-bval) > tol)
      {
        if (aval < bval)
          return true;

        return false;
      }
    }

    return false;
  }

  bool operator>(const eoParetoOneConstraintFitness<FitnessTraits>& _other) const
  {
    return _other < *this;
  }

  bool operator<=(const eoParetoOneConstraintFitness<FitnessTraits>& _other) const
  {
    return operator==(_other) || operator<(_other);
  }

  bool operator>=(const eoParetoOneConstraintFitness<FitnessTraits>& _other) const
  {
    return _other <= *this;
  }

  bool operator==(const eoParetoOneConstraintFitness<FitnessTraits>& _other) const
  { // check if they're all within tolerance
    for (unsigned i = 0; i < size(); ++i)
    {
      if (fabs(operator[](i) - _other[i]) > FitnessTraits::tol())
      {
        return false;
      }
    }
    return true;
  }

  bool operator!=(const eoParetoOneConstraintFitness<FitnessTraits>& _other) const
  { return ! operator==(_other); }

};

template <class FitnessTraits>
std::ostream& operator<<(std::ostream& os, const eoParetoOneConstraintFitness<FitnessTraits>& fitness)
{
  for (unsigned i = 0; i < fitness.size(); ++i)
  {
    os << fitness[i] << ' ';
  }
  os << fitness.ConstraintValue() << " " ;
  return os;
}

template <class FitnessTraits>
std::istream& operator>>(std::istream& is, eoParetoOneConstraintFitness<FitnessTraits>& fitness)
{
  fitness = eoParetoOneConstraintFitness<FitnessTraits>();
  for (unsigned i = 0; i < fitness.size(); ++i)
  {
    is >> fitness[i];
  }
  double r;
  is >> r;
  fitness.ConstraintValue(r);
  return is;
}

#endif
