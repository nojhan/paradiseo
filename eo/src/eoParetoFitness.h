// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoParetoFitness.h
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

    Contact: mak@dhi.dk
             Marc.Schoenauer@polytechnique.fr
 */
//-----------------------------------------------------------------------------

#ifndef _eoParetoFitness_h
#define _eoParetoFitness_h

#include <vector>

/**
  eoFitnessTraits: a traits class to specify the number of objectives and which one are maximizing
  or not. See eoParetoFitness for its use. If you define your own, make sure you make the functions
  static!
*/
class eoParetoFitnessTraits
{
  public :

  static unsigned nObjectives()          { return 2; }
  static double tol()               { return 1e-6; }
  static bool maximizing(int which) { return true; } // by default: all are maximizing
};

/**
  eoParetoFitness class: vector of doubles with overloaded comparison operators. Comparison is done
  on pareto dominance. The template argument FitnessTraits defaults to eoParetoFitnessTraits, which
  can be replaces at will by any other class that implements the static functions defined therein.

  Note that the comparison defines a partial order, so that
    !(a < b) && !(b <a) does not neccessarily imply that (a==b)
  The other way around does hold.
*/
template <class FitnessTraits = eoParetoFitnessTraits>
class eoParetoFitness : public std::vector<double>
{
public :

  typedef FitnessTraits fitness_traits;

  eoParetoFitness(void) : std::vector<double>(FitnessTraits::nObjectives(),0.0) {}

  /// Partial order based on Pareto-dominance
  //bool operator<(const eoParetoFitness<FitnessTraits>& _other) const
  bool dominates(const eoParetoFitness<FitnessTraits>& _other) const
  {
    bool dom = false;

    double tol = FitnessTraits::tol();
    const vector<double>& performance = *this;
    const vector<double>& otherperformance = _other;

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

    return dom;
  }

  /// compare *not* on dominance, but on the first, then the second, etc
  bool operator<(const eoParetoFitness<FitnessTraits>& _other) const
  {
    double tol = FitnessTraits::tol();
    const vector<double>& performance = *this;
    const vector<double>& otherperformance = _other;
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

  bool operator>(const eoParetoFitness<FitnessTraits>& _other) const
  {
    return _other < *this;
  }

  bool operator<=(const eoParetoFitness<FitnessTraits>& _other) const
  {
    return operator==(_other) || operator<(_other);
  }

  bool operator>=(const eoParetoFitness<FitnessTraits>& _other) const
  {
    return _other <= *this;
  }

  bool operator==(const eoParetoFitness<FitnessTraits>& _other) const
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

  bool operator!=(const eoParetoFitness<FitnessTraits>& _other) const
  { return ! operator==(_other); }

};

template <class FitnessTraits>
std::ostream& operator<<(std::ostream& os, const eoParetoFitness<FitnessTraits>& fitness)
{
  for (unsigned i = 0; i < fitness.size(); ++i)
  {
    os << fitness[i] << ' ';
  }
  return os;
}

template <class FitnessTraits>
std::istream& operator>>(std::istream& is, eoParetoFitness<FitnessTraits>& fitness)
{
  fitness = eoParetoFitness<FitnessTraits>();
  for (unsigned i = 0; i < fitness.size(); ++i)
  {
    is >> fitness[i];
  }
  return is;
}

#endif
