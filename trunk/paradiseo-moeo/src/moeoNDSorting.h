// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

/* -----------------------------------------------------------------------------
   moeoNDSorting.h
   (c) Deneche Abdelhakim, 2006

    Contact: paradiseo-help@lists.gforge.inria.fr 
*/
//-----------------------------------------------------------------------------

#ifndef moeoNDSorting_h
#define moeoNDSorting_h

#include <cfloat>
#include <eoNDSorting.h>

# define INF 1.0e14 // DBL_MAX

/** @brief Fast Elitist Non-Dominant Sorting Genetic Algorithm

	Note : This is a corrected version of the original eoNDSorting_II class
	.
    @see eoNDSorting_II
*/
template <class EOT>
class moeoNDSorting_II : public eoNDSorting<EOT>
{
  public:

    moeoNDSorting_II(bool nasty_flag_ = false) : eoNDSorting<EOT>(nasty_flag_) {}

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

      // set boundary at INF (so it will get chosen over all the others
      niche_count[performance[0].second] = INF;
      niche_count[performance.back().second] = INF;

 	  if (performance[0].first != performance.back().first)
 	  {
       	for (i = 1; i < _cf.size()-1; ++i)
       	{ 	
		  if (niche_count[performance[i].second] != INF)
		  {
			  niche_count[performance[i].second] += (performance[i+1].first - performance[i-1].first)/
	  			(performance.back().first-performance[0].first);
		  }
       	}
 	  }
    }

	// transform niche_count into penality
	for (i = 0; i < niche_count.size(); ++i)
    {
		niche_count[i] = INF - niche_count[i];
	}
	
    return niche_count;
  }
};

#endif
