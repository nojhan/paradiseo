#ifndef eoNSGA_II_Eval_h
#define eoNSGA_II_Eval_h

#include <moo/eoFrontSorter.h>
#include <moo/eoMOEval.h>

/** @brief Fast Elitist Non-Dominant Sorting Genetic Algorithm

  Adapted from Deb, Agrawal, Pratab and Meyarivan: A Fast Elitist
  Non-Dominant Sorting Genetic Algorithm for MultiObjective
  Optimization: NSGA-II KanGAL Report No. 200001

*/
template <class EOT>
class eoNSGA_II_Eval : public eoMOEval<EOT>
{
  public:

  eoNSGA_II_Eval(eoEvalFunc<EOT>& eval)    : eoMOEval<EOT>(eval) {}
  eoNSGA_II_Eval(eoPopEvalFunc<EOT>& eval) : eoMOEval<EOT>(eval) {}
    
    virtual void operator()(eoPop<EOT>& parents, eoPop<EOT>& offspring) {
        eval(parents, offspring);

        std::vector<EOT*> pop;
        pop.reserve(parents.size() + offspring.size());
        for (unsigned i = 0; i < parents.size(); ++i) pop.push_back(&parents[i]);
        for (unsigned i = 0; i < offspring.size(); ++i) pop.push_back(&offspring[i]);

        typename eoFrontSorter<EOT>::front_t front = sorter(pop);

        for (unsigned i = 0; i < front.size(); ++i) {
            assign_worths(front[i], front.size() - i, pop); 
        }
    }


  

  eoFrontSorter<EOT> sorter;
  
  private:
  typedef std::pair<double, unsigned> double_index_pair;

  class compare_nodes
  {
  public :
    bool operator()(const double_index_pair& a, const double_index_pair& b) const
    {
      return a.first < b.first;
    }
  };

  void assign_worths(const std::vector<detail::FitnessInfo>& front, unsigned rank, std::vector<EOT*>& parents) {
  
    typedef typename EOT::Fitness::fitness_traits traits;
    unsigned i;

    unsigned nObjectives = traits::nObjectives(); //_pop[_cf[0]].fitness().size();
    
    std::vector<double> niche_distance(front.size());

    for (unsigned o = 0; o < nObjectives; ++o)
    {

      std::vector<std::pair<double, unsigned> > performance(front.size());
      for (i =0; i < front.size(); ++i)
      {
        performance[i].first = front[i].fitness[o];
        performance[i].second = i;
      }

      std::sort(performance.begin(), performance.end(), compare_nodes()); // a lambda operator would've been nice here

      std::vector<double> nc(front.size(), 0.0);

      for (i = 1; i < front.size()-1; ++i)
      { // and yet another level of indirection
        nc[performance[i].second] = performance[i+1].first - performance[i-1].first;
      }

      // set boundary at max_dist + 1 (so it will get chosen over all the others
      //nc[performance[0].second]     += 0;
      nc[performance.back().second] += std::numeric_limits<double>::infinity(); // best on objective
      
      for (i = 0; i < nc.size(); ++i)
      {
        niche_distance[i] += nc[i];
      }
    }
    
    // now we've got niche_distances, scale them between (0, 1), making sure that infinities get maximum rank
    
    double max = 0;
    for (unsigned i = 0; i < niche_distance[i]; ++i) {
        if (niche_distance[i] != std::numeric_limits<double>::infinity()) {
            max = std::max(max, niche_distance[i]);
        }
    }
    
    for (unsigned i = 0; i < front.size(); ++i) {
        double dist = niche_distance[i];
        if (dist == std::numeric_limits<double>::infinity()) {
            dist = 1.0;
        } else {
            dist /= (1+max);
        }
        
        unsigned idx = front[i].index;
        
        typename EOT::Fitness f = parents[idx]->fitness();
        f.setWorth(rank + dist);
        //std::cout << "Base rank " << rank << " dist " << dist << " result " << (rank+dist) << std::endl;
         
        parents[idx]->fitness(f);
    }

  }
};

#endif
