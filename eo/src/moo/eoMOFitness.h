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
#include <limits>

/**
 * eoMOFitnessTraits: a traits class to specify 
 *           the number of objectives and which one are direction or not
 * See test/t-eoParetoFitness for its use. 
 *
 * If you define your own, make sure you make the functions static!
*/
class eoMOFitnessTraits
{
  public :

  /// Number of Objectives
  static unsigned nObjectives()          { return 2; }
  
  /// by default: all are maximizing, zero will lead to ignored fitness, negative to minimization for that objective
  static double direction(unsigned which)    { return 1; } 
  
  /// tolerance for dominance check (if within this tolerance objective values are considered equal)
  static double tol() { return 1e-6; } 
    
};

namespace dominance {
    
    template <class Traits>
    inline double worst_possible_value(unsigned objective) {
        double dir = Traits::direction(objective);
        if (dir == 0.) return 0.0;
        if (dir < 0.) return std::numeric_limits<double>::infinity();
        return -std::numeric_limits<double>::infinity();
    }

    template <class Traits>
    inline double best_possible_value(unsigned objective) {
        return -worst_possible_value<Traits>(objective);
    }

    enum dominance_result { non_dominated_equal, first, second, non_dominated };

    template <class FitnessDirectionTraits>
    inline dominance_result check(const std::vector<double>& p1, const std::vector<double>& p2, double tolerance) {
        
        bool all_equal = true;
        bool a_better_in_one = false;
        bool b_better_in_one = false;

        for (unsigned i = 0; i < p1.size(); ++i) {
            
            double maxim = FitnessDirectionTraits::direction(i);
            double aval = maxim * p1[i];
            double bval = maxim * p2[i];
            
            if ( fabs(aval-bval) > tolerance ) {
                all_equal = false;
                if (aval > bval) {
                    a_better_in_one = true;

                } else {
                    b_better_in_one = true;
                }
                // check if we need to go on

                if (a_better_in_one && b_better_in_one) return non_dominated;
            }

        }
            
        if (all_equal) return non_dominated_equal;
         
        if (a_better_in_one) return first;
        // else b dominates a (check for non-dominance done above
        return second;
    }

    template <class IntType>
    inline dominance_result check_discrete(const std::vector<IntType>& a, const std::vector<IntType>& b) {

        bool all_equal = true;
        bool a_better_in_one = false;
        bool b_better_in_one = false;

        for (unsigned i = 0; i < a.size(); ++i) {
        
            if ( a[i] != b[i] ) {
                all_equal = false;
                if (a[i] > b[i]) {
                    a_better_in_one = true;

                } else {
                    b_better_in_one = true;
                }
                // check if we need to go on

                if (a_better_in_one && b_better_in_one) return non_dominated;
            }

        }
        
        if (all_equal) return non_dominated_equal;
     
        if (a_better_in_one) return first;
        // else b dominates a (check for non-dominance done above
        return second;
    }
    
    template <class FitnessTraits>
    inline dominance_result check(const std::vector<double>& p1, const std::vector<double>& p2) {
        return check<FitnessTraits>(p1, p2, FitnessTraits::tol());
    }
    
     

} // namespace dominance

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
    
  eoMOFitness() : std::vector<double>(FitnessTraits::nObjectives()), worthValid(false) { reset(); }
  
  explicit eoMOFitness(double def) : std::vector<double>(FitnessTraits::nObjectives(),def), worthValid(false) {}

  // Ctr from a std::vector<double>
  explicit eoMOFitness(std::vector<double> & _v) : std::vector<double>(_v), worthValid(false) {}

  virtual ~eoMOFitness() {} // in case someone wants to subclass
  eoMOFitness(const eoMOFitness<FitnessTraits>& org) { operator=(org); }

  eoMOFitness<FitnessTraits>& operator=(const eoMOFitness<FitnessTraits>& org) {
        
        std::vector<double>::operator=(org);
        worth = org.worth;
        worthValid = org.worthValid;

        return *this;
  }

  void reset() {

    for (unsigned i = 0; i < size(); ++i) {
        this->operator[](i) = dominance::worst_possible_value<FitnessTraits>(i);
    }
  }

  // Make the objective 'feasible' by setting it to the best possible value
  void setFeasible(unsigned objective) { this->operator[](objective) = dominance::best_possible_value<FitnessTraits>(objective); }

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
  
  /// Check on dominance: returns 0 if non-dominated, 1 if this dominates other, -1 if other dominates this
  int check_dominance(const eoMOFitness<FitnessTraits>& _other) const
  {
    dominance::dominance_result dom = dominance::check<FitnessTraits>(*this, _other);

    return dom == dominance::first? 1 : (dom == dominance::second? -1 : 0);
  }

  /// normalized fitness: all maximizing, removed the irrelevant ones
  std::vector<double> normalized() const {
        std::vector<double> f;
        
        for (unsigned j = 0; j < FitnessTraits::nObjectives(); ++j) {
            if (FitnessTraits::direction(j) != 0) {
                f.push_back( FitnessTraits::direction(j) * this->operator[](j));
            }
        }
        
        return f;
  }

  /// returns true if this dominates other
  bool dominates(const eoMOFitness<FitnessTraits>& _other) const
  {
    return check_dominance(_other) == 1;
  }

  /// compare *not* on dominance, but on worth 
  bool operator<(const eoMOFitness<FitnessTraits>& _other) const
  {
    return getWorth() < _other.getWorth();
  }

  bool operator>(const eoMOFitness<FitnessTraits>& _other) const
  {
    return _other > *this;
  }

  bool operator<=(const eoMOFitness<FitnessTraits>& _other) const
  {
    return getWorth() <= _other.getWorth();
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

namespace dominance {
    template <class FitnessTraits>
    inline dominance_result check(const eoMOFitness<FitnessTraits>& p1, const eoMOFitness<FitnessTraits>& p2) {
        return check<FitnessTraits>(p1, p2, FitnessTraits::tol());   
    }
}

#endif
