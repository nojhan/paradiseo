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
#include <eoDominanceMap.h>

/**
  Non dominated sorting, it *is a* vector of doubles, the integer part is the rank (to which front it belongs),
  the fractional part the niching penalty or distance penalty or whatever penalty you want to squeeze into
  the bits.
*/
template <class EOT>
class eoNDSorting : public eoPerf2Worth<EOT, double>
{
  public :

    eoNDSorting(eoDominanceMap<EOT>& _dominanceMap) :
      eoPerf2Worth<EOT, double>(), dominanceMap(_dominanceMap) {}

    /** Pure virtual function that calculates the 'distance' for each element to the current front
        Implement to create your own nondominated sorting algorithm. The size of the returned vector
        should be equal to the size of the current_front.
    */
    virtual vector<double> niche_penalty(const vector<unsigned>& current_front, const eoPop<EOT>& _pop) = 0;

    /// Do the work
    void operator()(const eoPop<EOT>& _pop)
    {
      dominanceMap(_pop);

      vector<bool> excluded(_pop.size(), false);

      value().resize(_pop.size());

      bool finished = false;

      int dominance_level = 0;

      while(!finished)
      {
        vector<double> ranks = dominanceMap.sum_dominators();

        vector<unsigned> current_front;
        current_front.reserve(_pop.size());

        finished = true;

        for (unsigned i = 0; i < _pop.size(); ++i)
        {
          if (excluded[i])
          {
            continue; // next please
          }

          if (ranks[i] < 1e-6)
          {// it's part of the current front
            excluded[i] = true;
            current_front.push_back(i);
            dominanceMap.remove(i); // remove from consideration
          }
          else
          {
            finished = false; // we need another go
          }
        }

        // Now we have the indices to the current front in current_front, do the niching
        vector<double> niche_count = niche_penalty(current_front, _pop);

        if (niche_count.size() != current_front.size())
        {
          throw logic_error("eoNDSorting: niche and front should have the same size");
        }

        double max_niche = *max_element(niche_count.begin(), niche_count.end());

        for (unsigned i = 0; i < current_front.size(); ++i)
        {
          value()[current_front[i]] = dominance_level + niche_count[i] / (max_niche + 1);
        }

        dominance_level++; // go to the next front
      }

      // now all that's left to do is to transform lower rank into higher worth
      double max_fitness = *std::max_element(value().begin(), value().end());

      for (unsigned i = 0; i < value().size(); ++i)
      {
        value()[i] = max_fitness - value()[i];
      }

    }

    const eoDominanceMap<EOT>& map() const;

  private :

  eoDominanceMap<EOT>& dominanceMap;
};

/**
  The original Non Dominated Sorting algorithm from Srinivas and Deb
*/
template <class EOT>
class eoNDSorting_I : public eoNDSorting<EOT>
{
public :
  eoNDSorting_I(eoDominanceMap<EOT>& _map, double _nicheSize) : eoNDSorting<EOT>(_map), nicheSize(_nicheSize) {}

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
  eoNDSorting_II(eoDominanceMap<EOT>& _map) : eoNDSorting<EOT>(_map) {}

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

      // set boundary penalty at 0 (so it will get chosen over all the others
      nc[performance[0].second] = 0;
      nc[performance.back().second] = 0;

      for (unsigned i = 0; i < nc.size(); ++i)
      {
        niche_count[i] += (max_dist + 1) - nc[i];
      }
    }

    return niche_count;
  }


};

#endif