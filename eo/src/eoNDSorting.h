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

#include <algorithm>
#include <eoPerf2Worth.h>

/**
  Non dominated sorting, it *is a* vector of doubles, the integer part is the rank (to which front it belongs),
  the fractional part the niching penalty or distance penalty or whatever penalty you want to squeeze into
  the bits.
*/
template <class EOT>
class eoNDSorting : public eoPerf2WorthCached<EOT, double>
{
  public :

    /** Pure virtual function that calculates the 'distance' for each element to the current front
        Implement to create your own nondominated sorting algorithm. The size of the returned vector
        should be equal to the size of the current_front.
    */
    virtual vector<double> niche_penalty(const vector<unsigned>& current_front, const eoPop<EOT>& _pop) = 0;

    /** implements fast nondominated sorting
    */
    class DummyEO : public EO<typename EOT::Fitness>
    {
      public: unsigned index;
    };

    void calculate_worths(const eoPop<EOT>& _pop)
    {
      value().resize(_pop.size());

      typedef typename EOT::Fitness::fitness_traits traits;

      if (traits::nObjectives() == 1)
      { // no need to do difficult sorting,

        eoPop<DummyEO> tmp_pop;
        tmp_pop.resize(_pop.size());

        // copy pop to dummy population (only need the fitnesses)
        for (unsigned i = 0; i < _pop.size(); ++i)
        {
          tmp_pop[i].fitness(_pop[i].fitness());
          tmp_pop[i].index = i;
        }

        // sort it
        tmp_pop.sort();

        //
        for (unsigned i = 0; i < _pop.size(); ++i)
        {
          value()[tmp_pop[i].index] = _pop.size() - i; // set rank
        }

        return;
      }

      vector<vector<unsigned> > S(_pop.size()); // which individuals does guy i dominate
      vector<unsigned> n(_pop.size(), 0); // how many individuals dominate guy i


      for (unsigned i = 0; i < _pop.size(); ++i)
      {
        for (unsigned j = 0; j < _pop.size(); ++j)
        {
          if (_pop[i].fitness().dominates(_pop[j].fitness()))
          { // i dominates j
            S[i].push_back(j); // add j to i's domination list

            //n[j]++; // as i dominates j
          }
          else if (_pop[j].fitness().dominates(_pop[i].fitness()))
          { // j dominates i, increment count for i, add i to the domination list of j
            n[i]++;

            //S[j].push_back(i);
          }
        }
      }

      vector<unsigned> current_front;
      current_front.reserve(_pop.size());

      // get the first front out
      for (unsigned i = 0; i < _pop.size(); ++i)
      {
        if (n[i] == 0)
        {
          current_front.push_back(i);
        }
      }

      vector<unsigned> next_front;
      next_front.reserve(_pop.size());

      unsigned front_index = 0; // which front are we processing
      while (!current_front.empty())
      {
        // Now we have the indices to the current front in current_front, do the niching
        vector<double> niche_count = niche_penalty(current_front, _pop);

        // Check whether the derived class was nice
        if (niche_count.size() != current_front.size())
        {
          throw logic_error("eoNDSorting: niche and front should have the same size");
        }

        double max_niche = *max_element(niche_count.begin(), niche_count.end());

        for (unsigned i = 0; i < current_front.size(); ++i)
        {
          value()[current_front[i]] = front_index + niche_count[i] / (max_niche + 1.); // divide by max_niche + 1 to ensure that this front does not overlap with the next
        }

        // Calculate which individuals are in the next front;

        for (unsigned i = 0; i < current_front.size(); ++i)
        {
          for (unsigned j = 0; j < S[current_front[i]].size(); ++j)
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

      // now all that's left to do is to transform lower rank into higher worth
      double max_fitness = *std::max_element(value().begin(), value().end());

      for (unsigned i = 0; i < value().size(); ++i)
      {
        value()[i] = max_fitness - value()[i];
        assert(n[i] == 0);
      }
    }
};

/**
  The original Non Dominated Sorting algorithm from Srinivas and Deb
*/
template <class EOT>
class eoNDSorting_I : public eoNDSorting<EOT>
{
public :
  eoNDSorting_I(double _nicheSize) : eoNDSorting<EOT>(), nicheSize(_nicheSize) {}

  vector<double> niche_penalty(const vector<unsigned>& current_front, const eoPop<EOT>& _pop)
  {
        vector<double> niche_count(current_front.size(), 0.);

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

/**
  Adapted from Deb, Agrawal, Pratab and Meyarivan: A Fast Elitist Non-Dominant Sorting Genetic Algorithm for MultiObjective Optimization: NSGA-II
  KanGAL Report No. 200001

  Note that this class does not do the sorting per se, but the sorting of it worth_vector will give the right order

  The crowding distance is calculated as the sum of the distances to the nearest neighbours. As we need to return the
  penalty value, we have to invert that and invert it again in the base class, but such is life, sigh
*/
template <class EOT>
class eoNDSorting_II : public eoNDSorting<EOT>
{
  public:

  typedef std::pair<double, unsigned> double_index_pair;

  class compare_nodes
  {
  public :
    bool operator()(const double_index_pair& a, const double_index_pair& b) const
    {
      return a.first < b.first;
    }
  };

  vector<double> niche_penalty(const vector<unsigned>& _cf, const eoPop<EOT>& _pop)
  {
    vector<double> niche_count(_cf.size(), 0.);

    unsigned nObjectives = _pop[_cf[0]].fitness().size();

    for (unsigned o = 0; o < nObjectives; ++o)
    {

      vector<pair<double, unsigned> > performance(_cf.size());
      for (unsigned i =0; i < _cf.size(); ++i)
      {
        performance[i].first = _pop[_cf[i]].fitness()[o];
        performance[i].second = i;
      }

      sort(performance.begin(), performance.end(), compare_nodes()); // a lambda operator would've been nice here

      vector<double> nc(niche_count.size(), 0.0);

      for (unsigned i = 1; i < _cf.size()-1; ++i)
      { // and yet another level of indirection
        nc[performance[i].second] = performance[i+1].first - performance[i-1].first;
      }

      double max_dist = *max_element(nc.begin(), nc.end());

      // set boundary at max_dist + 1 (so it will get chosen over all the others
      nc[performance[0].second] = max_dist + 1;
      nc[performance.back().second] = max_dist + 1;

      for (unsigned i = 0; i < nc.size(); ++i)
      {
        niche_count[i] += (max_dist + 1) - nc[i];
      }
    }

    return niche_count;
  }


};

#endif