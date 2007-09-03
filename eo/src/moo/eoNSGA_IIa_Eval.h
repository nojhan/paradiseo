#ifndef eoNSGA_IIa_Eval_h
#define eoNSGA_IIa_Eval_h

#include <moo/eoFrontSorter.h>
#include <moo/eoMOEval.h>
#include <cassert>

/** @brief Fast Elitist Non-Dominant Sorting Genetic Algorithm
    
    Variant of the NSGA-II, where the ranking is based on a top-down distance based mechanism ( O(n^2)! )

*/

namespace nsga2a {
    extern unsigned assign_worths(std::vector<detail::FitnessInfo> front, unsigned rank, std::vector<double>& worths);
}

template <class EOT>
class eoNSGA_IIa_Eval : public eoMOEval<EOT>
{
  public:
  
  eoNSGA_IIa_Eval(eoEvalFunc<EOT>& eval)    : eoMOEval<EOT>(eval) {}
  eoNSGA_IIa_Eval(eoPopEvalFunc<EOT>& eval) : eoMOEval<EOT>(eval) {}
  

  virtual void operator()(eoPop<EOT>& parents, eoPop<EOT>& offspring) {
        eval(parents, offspring);

        std::vector<EOT*> pop;
        pop.reserve(parents.size() + offspring.size());
        for (unsigned i = 0; i < parents.size(); ++i) pop.push_back(&parents[i]);
        for (unsigned i = 0; i < offspring.size(); ++i) {
            pop.push_back(&offspring[i]);
        }

        typename eoFrontSorter<EOT>::front_t front = sorter(pop);
        
        std::vector<double> worths(pop.size());      
        unsigned rank = pop.size();
        for (unsigned i = 0; i < front.size(); ++i) {
            rank = nsga2a::assign_worths(front[i], rank, worths); 
        }
    
        for (unsigned i = 0; i < worths.size(); ++i) {
            typename EOT::Fitness f = pop[i]->fitness();
            f.setWorth(worths[i]);
            pop[i]->fitness(f);
        }
  }

  private:

  eoFrontSorter<EOT> sorter;
 
  // implementation
};

#endif
