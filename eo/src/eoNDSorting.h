/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

   -----------------------------------------------------------------------------
   eoNDSorting.h
   (c) Maarten Keijzer, Marc Schoenauer, 2001

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
             mkeijzer@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef eoNDSorting_h
#define eoNDSorting_h

#include <EO.h>
#include <algorithm>
#include <eoPop.h>
#include <eoPerf2Worth.h>
#include <cassert>

/**
  Non dominated sorting, it *is a* std::vector of doubles, the integer part is the rank (to which front it belongs),
  the fractional part the niching penalty or distance penalty or whatever penalty you want to squeeze into
  the bits.
*/
template <class EOT>
class eoNDSorting : public eoPerf2WorthCached<EOT, double>
{
  public :

    using eoPerf2WorthCached<EOT, double>::value;
      eoNDSorting(bool nasty_flag_ = false)
          : nasty_declone_flag_that_only_is_implemented_for_two_objectives(nasty_flag_)
        {}


    eoNDSorting()
        : nasty_declone_flag_that_only_is_implemented_for_two_objectives(false)
        {}

    /** Pure virtual function that calculates the 'distance' for each element in the current front
        Implement to create your own nondominated sorting algorithm. The size of the returned std::vector
        should be equal to the size of the current_front.
    */
    virtual std::vector<double> niche_penalty(const std::vector<unsigned>& current_front, const eoPop<EOT>& _pop) = 0;

    void calculate_worths(const eoPop<EOT>& _pop)
    {
        // resize the worths beforehand
        value().resize(_pop.size());

        typedef typename EOT::Fitness::fitness_traits traits;

        switch (traits::nObjectives())
        {
            case 1:
                {
                    one_objective(_pop);
                    return;
                }
            case 2:
                {
                    two_objectives(_pop);
                    return;
                }
            default :
                {
                    m_objectives(_pop);
                }
        }
    }

private :

    /** used in fast nondominated sorting
        DummyEO is just a storage place for fitnesses and
        to store the original index
    */
    class DummyEO : public EO<typename EOT::Fitness>
    {
      public: unsigned index;
    };

    void one_objective(const eoPop<EOT>& _pop)
    {
        unsigned i;
        std::vector<DummyEO> tmp_pop;
        tmp_pop.resize(_pop.size());

        // copy pop to dummy population (only need the fitnesses)
        for (i = 0; i < _pop.size(); ++i)
        {
          tmp_pop[i].fitness(_pop[i].fitness());
          tmp_pop[i].index = i;
        }

        std::sort(tmp_pop.begin(), tmp_pop.end(), std::greater<DummyEO>());

        for (i = 0; i < _pop.size(); ++i)
        {
          value()[tmp_pop[i].index] = _pop.size() - i; // set rank
        }

        // no point in calculcating niche penalty, as every distinct fitness value has a distinct rank
    }

    /**
     * Optimization for two objectives. Makes the algorithm run in
     * complexity O(n log n) where n is the population size
     *
     * This is the same complexity as for a single objective
     * or truncation selection or sorting.
     *
     * It will perform a sort on the two objectives seperately,
     * and from the information on the ranks of the individuals on
     * these two objectives, the non-dominated sorting rank is determined.
     * There are then three nlogn operations in place: one sort per objective,
     * plus a binary search procedure to combine the information about the
     * ranks.
     *
     * After that it is a simple exercise to calculate the distance
     * penalty
     */

    void two_objectives(const eoPop<EOT>& _pop)
    {
                unsigned i;
        typedef typename EOT::Fitness::fitness_traits traits;
        assert(traits::nObjectives() == 2);

        std::vector<unsigned> sort1(_pop.size()); // index into population sorted on first objective

        for (i = 0; i < _pop.size(); ++i)
        {
            sort1[i] = i;
        }

        std::sort(sort1.begin(), sort1.end(), Sorter(_pop));

        // Ok, now the meat of the algorithm

        unsigned last_front = 0;

        double max1 = -1e+20;
        for (i = 0; i < _pop.size(); ++i)
        {
            max1 = std::max(max1, _pop[i].fitness()[1]);
        }

        max1 = max1 + 1.0; // add a bit to it so that it is a real upperbound

        unsigned prev_front = 0;
        std::vector<double> d;
        d.resize(_pop.size(), max1); // initialize with the value max1 everywhere

        std::vector<std::vector<unsigned> > fronts(_pop.size()); // to store indices into the front

        for (i = 0; i < _pop.size(); ++i)
        {
            unsigned index = sort1[i];

            // check for clones and delete them
            if (i > 0)
            {
                unsigned prev = sort1[i-1];
                if ( _pop[index].fitness() == _pop[prev].fitness())
                { // it's a clone, give it the worst rank!

                    if (nasty_declone_flag_that_only_is_implemented_for_two_objectives)
                        //declone
                        fronts.back().push_back(index);
                    else // assign it the rank of the previous
                        fronts[prev_front].push_back(index);
                    continue;
                }
            }

            double value2 = _pop[index].fitness()[1];

            if (traits::maximizing(1))
                value2 = max1 - value2;

            // perform binary search using std::upper_bound, a log n operation for each member
            std::vector<double>::iterator it =
                std::upper_bound(d.begin(), d.begin() + last_front, value2);

            unsigned front = unsigned(it - d.begin());
            if (front == last_front) ++last_front;

            assert(it != d.end());

            *it = value2; //update d
            fronts[front].push_back(index); // add it to the front

            prev_front = front;
        }

        // ok, and finally the niche penalty

        for (i = 0; i < fronts.size(); ++i)
        {
            if (fronts[i].size() == 0) continue;

            // Now we have the indices to the current front in current_front, do the niching
            std::vector<double> niche_count = niche_penalty(fronts[i], _pop);

            // Check whether the derived class was nice
            if (niche_count.size() != fronts[i].size())
            {
              throw std::logic_error("eoNDSorting: niche and front should have the same size");
            }

            double max_niche = *std::max_element(niche_count.begin(), niche_count.end());

            for (unsigned j = 0; j < fronts[i].size(); ++j)
            {
              value()[fronts[i][j]] = i + niche_count[j] / (max_niche + 1.); // divide by max_niche + 1 to ensure that this front does not overlap with the next
            }

        }

        // invert ranks to obtain a 'bigger is better' score
        rank_to_worth();
    }

    class Sorter
    {
        public:
            Sorter(const eoPop<EOT>& _pop) : pop(_pop) {}

            bool operator()(unsigned i, unsigned j) const
            {
                typedef typename EOT::Fitness::fitness_traits traits;

                double diff = pop[i].fitness()[0] - pop[j].fitness()[0];

                if (fabs(diff) < traits::tol())
                {
                    diff = pop[i].fitness()[1] - pop[j].fitness()[1];

                    if (fabs(diff) < traits::tol())
                        return false;

                    if (traits::maximizing(1))
                        return diff > 0.;
                    return diff < 0.;
                }

                if (traits::maximizing(0))
                    return diff > 0.;
                return diff < 0.;
            }

            const eoPop<EOT>& pop;
    };

    void m_objectives(const eoPop<EOT>& _pop)
    {
      unsigned i;

      typedef typename EOT::Fitness::fitness_traits traits;

      std::vector<std::vector<unsigned> > S(_pop.size()); // which individuals does guy i dominate
      std::vector<unsigned> n(_pop.size(), 0); // how many individuals dominate guy i

      unsigned j;
      for (i = 0; i < _pop.size(); ++i)
      {
        for (j = 0; j < _pop.size(); ++j)
        {
          if (_pop[i].fitness().dominates(_pop[j].fitness()))
          { // i dominates j
            S[i].push_back(j); // add j to i's domination std::list

            //n[j]++; // as i dominates j
          }
          else if (_pop[j].fitness().dominates(_pop[i].fitness()))
          { // j dominates i, increment count for i, add i to the domination std::list of j
            n[i]++;

            //S[j].push_back(i);
          }
        }
      }

      std::vector<unsigned> current_front;
      current_front.reserve(_pop.size());

      // get the first front out
      for (i = 0; i < _pop.size(); ++i)
      {
        if (n[i] == 0)
        {
          current_front.push_back(i);
        }
      }

      std::vector<unsigned> next_front;
      next_front.reserve(_pop.size());

      unsigned front_index = 0; // which front are we processing
      while (!current_front.empty())
      {
        // Now we have the indices to the current front in current_front, do the niching
        std::vector<double> niche_count = niche_penalty(current_front, _pop);

        // Check whether the derived class was nice
        if (niche_count.size() != current_front.size())
        {
          throw std::logic_error("eoNDSorting: niche and front should have the same size");
        }

        double max_niche = *std::max_element(niche_count.begin(), niche_count.end());

        for (i = 0; i < current_front.size(); ++i)
        {
          value()[current_front[i]] = front_index + niche_count[i] / (max_niche + 1.); // divide by max_niche + 1 to ensure that this front does not overlap with the next
        }

        // Calculate which individuals are in the next front;

        for (i = 0; i < current_front.size(); ++i)
        {
          for (j = 0; j < S[current_front[i]].size(); ++j)
          {
            unsigned dominated_individual = S[current_front[i]][j];
            n[dominated_individual]--; // As we remove individual i -- being part of the current front -- it no longer dominates j

            if (n[dominated_individual] == 0) // it should be in the current front
            {
              next_front.push_back(dominated_individual);
            }
          }
        }

        front_index++; // go to the next front
        swap(current_front, next_front); // make the next front current
        next_front.clear(); // clear it for the next iteration
      }

      rank_to_worth();
    }

    void rank_to_worth()
    {
      // now all that's left to do is to transform lower rank into higher worth
      double max_fitness = *std::max_element(value().begin(), value().end());

      // but make sure it's an integer upper bound, so that all ranks inside the highest integer are the front
      max_fitness = ceil(max_fitness);

      for (unsigned i = 0; i < value().size(); ++i)
      {
        value()[i] = max_fitness - value()[i];
      }

    }
    public : bool nasty_declone_flag_that_only_is_implemented_for_two_objectives;
};

/**
  The original Non Dominated Sorting algorithm from Srinivas and Deb
*/
template <class EOT>
class eoNDSorting_I : public eoNDSorting<EOT>
{
public :
  eoNDSorting_I(double _nicheSize, bool nasty_flag_ = false) : eoNDSorting<EOT>(nasty_flag_), nicheSize(_nicheSize) {}

  std::vector<double> niche_penalty(const std::vector<unsigned>& current_front, const eoPop<EOT>& _pop)
  {
        std::vector<double> niche_count(current_front.size(), 0.);

        for (unsigned i = 0; i < current_front.size(); ++i)
        { // calculate whether the other points lie within the nice
          for (unsigned j = 0; j < current_front.size(); ++j)
          {
            if (i == j)
              continue;

            double dist = 0.0;

            for (unsigned k = 0; k < _pop[current_front[j]].fitness().size(); ++k)
            {
              double d = _pop[current_front[i]].fitness()[k] - _pop[current_front[j]].fitness()[k];
              dist += d*d;
            }

            if (dist < nicheSize)
            {
              niche_count[i] += 1.0 - pow(dist / nicheSize,2.);
            }
          }
        }

        return niche_count;
  }

  private :

  double nicheSize;
};

/** @brief Fast Elitist Non-Dominant Sorting Genetic Algorithm

  Adapted from Deb, Agrawal, Pratab and Meyarivan: A Fast Elitist
  Non-Dominant Sorting Genetic Algorithm for MultiObjective
  Optimization: NSGA-II KanGAL Report No. 200001

  Note that this class does not do the sorting per se, but the sorting
  of it worth_std::vector will give the right order

  The crowding distance is calculated as the sum of the distances to
  the nearest neighbours. As we need to return the penalty value, we
  have to invert that and invert it again in the base class, but such
  is life, sigh
*/
template <class EOT>
class eoNDSorting_II : public eoNDSorting<EOT>
{
  public:

    eoNDSorting_II(bool nasty_flag_ = false) : eoNDSorting<EOT>(nasty_flag_) {}

  typedef std::pair<double, unsigned> double_index_pair;

  class compare_nodes
  {
  public :
    bool operator()(const double_index_pair& a, const double_index_pair& b) const
    {
      return a.first < b.first;
    }
  };

  /// _cf points into the elements that consist of the current front
  std::vector<double> niche_penalty(const std::vector<unsigned>& _cf, const eoPop<EOT>& _pop)
  {
    typedef typename EOT::Fitness::fitness_traits traits;
    unsigned i;
    std::vector<double> niche_count(_cf.size(), 0.);


    unsigned nObjectives = traits::nObjectives(); //_pop[_cf[0]].fitness().size();

    for (unsigned o = 0; o < nObjectives; ++o)
    {

      std::vector<std::pair<double, unsigned> > performance(_cf.size());
      for (i =0; i < _cf.size(); ++i)
      {
        performance[i].first = _pop[_cf[i]].fitness()[o];
        performance[i].second = i;
      }

      std::sort(performance.begin(), performance.end(), compare_nodes()); // a lambda operator would've been nice here

      std::vector<double> nc(niche_count.size(), 0.0);

      for (i = 1; i < _cf.size()-1; ++i)
      { // calculate distance
        nc[performance[i].second] = performance[i+1].first - performance[i-1].first;
      }

      double max_dist = *std::max_element(nc.begin(), nc.end());

      // set boundary at max_dist + 1 (so it will get chosen over all the others
      nc[performance[0].second]     += max_dist + 1;
      nc[performance.back().second] += max_dist + 1;

      for (i = 0; i < nc.size(); ++i)
      {
        niche_count[i] += (max_dist + 1 - nc[i]);
      }
    }

    return niche_count;
  }
};

#endif
