// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoMOFitness.h
// (c) Maarten Keijzer and Marc Schoenauer, 2007
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
             Marc.Schoenauer@inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef _eoMOFitness_h
#define _eoMOFitness_h

#include <math.h>
#include <vector>
#include <stdexcept>
#include <iostream>


/**
 * eoMOFitnessTraits: a traits class to specify 
 *           the number of objectives and which one are maximizing or not
 * See test/t-eoParetoFitness for its use. 
 *
 * If you define your own, make sure you make the functions static!
*/
class eoMOFitnessTraits
{
  public :

  static unsigned nObjectives()          { return 2; }
  static double maximizing(int which) { return 1; } // by default: all are maximizing, zero will lead to ignored fitness, negative minimizes
  static double tol() { return 1e-6; } // tolerance for distance calculations
};

/**
  eoMOFitness class: std::vector of doubles with overloaded comparison operators. Comparison is done
  on 'worth'. This worth needs to be set elsewhere. The template argument FitnessTraits defaults to eMOFitnessTraits, which
  can be replaces at will by any other class that implements the static functions defined therein.

  Note that the comparison defines a partial order, so that
    !(a < b) && !(b <a) does not neccessarily imply that (a==b)
  The other way around does hold.
*/
template <class FitnessTraits = eoMOFitnessTraits>
class eoMOFitness : public std::vector<double>
{
  double worth; // used for sorting and selection, by definition, bigger is better
  bool worthValid;
public :

  typedef FitnessTraits fitness_traits;
    
  eoMOFitness(double def = 0.0) : std::vector<double>(FitnessTraits::nObjectives(),def), worthValid(false) {}

  // Ctr from a std::vector<double>
  eoMOFitness(std::vector<double> & _v) : std::vector<double>(_v), worthValid(false) {}

  /** access to the traits characteristics (so you don't have to write 
   * a lot of typedef's around
   */
  static void setUp(unsigned _n, std::vector<bool> & _b) {FitnessTraits::setUp(_n, _b);}
  static bool maximizing(unsigned _i) { return FitnessTraits::maximizing(_i);}

  void setWorth(double worth_) {
        worth = worth_;
        worthValid = true;
  }

  double getWorth() const {
        if (!worthValid) {
           throw std::runtime_error("invalid worth"); 
        }
        return worth;
  }

  bool validWorth() const { return worthValid; }
  void invalidateWorth() { worthValid = false; }

  /// Partial order based on Pareto-dominance
  //bool operator<(const eoMOFitness<FitnessTraits>& _other) const
  bool dominates(const eoMOFitness<FitnessTraits>& _other) const
  {
    bool dom = false;

    const std::vector<double>& performance = *this;
    const std::vector<double>& otherperformance = _other;

    for (unsigned i = 0; i < FitnessTraits::nObjectives(); ++i)
    {
      double maxim = FitnessTraits::maximizing(i);
      double aval = maxim * performance[i];
      double bval = maxim * otherperformance[i];

      if (fabs(aval-bval)>FitnessTraits::tol)
      {
        if (aval < bval)
        {
          return false; // cannot dominate
        }
        // else aval > bval
        dom = true; // for the moment: goto next objective
      }
      //else they're equal in this objective, goto next
    }

    return dom;
  }

  /// compare *not* on dominance, but on worth 
  bool operator<(const eoMOFitness<FitnessTraits>& _other) const
  {
    return getWorth() > _other.getWorth();
  }

  bool operator>(const eoMOFitness<FitnessTraits>& _other) const
  {
    return _other < *this;
  }

  bool operator<=(const eoMOFitness<FitnessTraits>& _other) const
  {
    return getWorth() >= _other.getWorth();
  }

  bool operator>=(const eoMOFitness<FitnessTraits>& _other) const
  {
    return _other <= *this;
  }

  bool operator==(const eoMOFitness<FitnessTraits>& _other) const
  { // check if they're all within tolerance
    return getWorth() == _other.getWorth();
  }

  bool operator!=(const eoMOFitness<FitnessTraits>& _other) const
  { return ! operator==(_other); }


    friend std::ostream& operator<<(std::ostream& os, const eoMOFitness<FitnessTraits>& fitness)
    {
      for (unsigned i = 0; i < fitness.size(); ++i)
      {
        os << fitness[i] << ' ';
      }
      os << fitness.worthValid << ' ' << fitness.worth;
      return os;
    }

    friend std::istream& operator>>(std::istream& is, eoMOFitness<FitnessTraits>& fitness)
    {
      fitness = eoMOFitness<FitnessTraits>();
      for (unsigned i = 0; i < fitness.size(); ++i)
      {
        is >> fitness[i];
      }
      is >> fitness.worthValid >> fitness.worth;
      return is;
    }

};

#endif
