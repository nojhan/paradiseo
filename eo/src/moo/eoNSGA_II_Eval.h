#ifndef eoNSGA_II_Eval_h
#define eoNSGA_II_Eval_h

#include <moo/eoFrontSorter.h>
#include <moo/eoMOEval.h>

/** @brief Fast Elitist Non-Dominant Sorting Genetic Algorithm

  Adapted from Deb, Agrawal, Pratab and Meyarivan: A Fast Elitist
  Non-Dominant Sorting Genetic Algorithm for MultiObjective
  Optimization: NSGA-II KanGAL Report No. 200001

*/

namespace nsga2 {  
  void assign_worths(const std::vector<detail::FitnessInfo>& front, unsigned rank, std::vector<double>& worths); 
}

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
        
        // calculate worths
        std::vector<double> worths(pop.size());
        for (unsigned i = 0; i < front.size(); ++i) {
            nsga2::assign_worths(front[i], front.size() - i, worths); 
        }
         
        // set worths
        for (unsigned i = 0; i < pop.size(); ++i) {
            pop[i]->fitnessReference().setWorth( worths[i]);
        }

    }
  

  eoFrontSorter<EOT> sorter;
  
  private:
  
};

#endif
