#ifndef eoNSGA_IIa_Eval_h
#define eoNSGA_IIa_Eval_h

#include <moo/eoFrontSorter.h>
#include <moo/eoMOEval.h>

/** @brief Fast Elitist Non-Dominant Sorting Genetic Algorithm
    
    Variant of the NSGA-II, where the ranking is based on a top-down distance based mechanism ( O(n^2)! )

*/
template <class EOT>
class eoNSGA_IIa_Eval : public eoMOEval<EOT>
{
  public:
  
  eoNSGA_IIa_Eval(eoEvalFunc<EOT>& eval)    : eoMOEval<EOT>(eval) {}
  eoNSGA_IIa_Eval(eoPopEvalFunc<EOT>& eval) : eoMOEval<EOT>(eval) {}
  

  void operator()(eoPop<EOT>& parents, eoPop<EOT>& offspring) {
        eval(parents, offspring);

        std::vector<EOT*> pop;
        pop.reserve(parents.size() + offspring.size());
        for (unsigned i = 0; i < parents.size(); ++i) pop.push_back(&parents[i]);
        for (unsigned i = 0; i < offspring.size(); ++i) pop.push_back(&offspring[i]);

        typename eoFrontSorter<EOT>::front_t front = sorter(pop);

        
        unsigned rank = parents.size();
        for (unsigned i = 0; i < front.size(); ++i) {
            rank = assign_worths(front[i], rank, pop); 
        }
  }

  private:

    eoFrontSorter<EOT> sorter;
 
  double distance(const std::vector<double>& f1, const std::vector<double>& f2, const std::vector<double>& range) {
        double dist = 0;
        for (unsigned i = 0; i < f1.size(); ++i) {
            double d = (f1[i] - f2[i])/range[i];
            dist += d*d;
        }
        return dist;
  }
  
  unsigned assign_worths(const std::vector<detail::FitnessInfo>& front, unsigned rank, std::vector<EOT*>& parents) {
    
    unsigned nDim = front[0].fitness.size();
    
    // find boundary points
    std::vector<unsigned> processed(nDim);
    
    for (unsigned i = 1; i < front.size(); ++i) {
        for (unsigned dim = 0; dim < nDim; ++dim) {
            if (front[i].fitness[dim] > front[processed[dim]].fitness[dim]) {
                processed[dim] = i;
            }
        }
    }
    
    // assign fitness to processed
    for (unsigned i = 0; i < processed.size(); ++i) {
        typename EOT::Fitness f = parents[ front[ processed[i] ].index]->fitness();
        f.setWorth(rank);
        parents[ front[ processed[i] ].index ]->fitness(f);   
    }
    rank--;

    // calculate ranges
    std::vector<double> mins(nDim, std::numeric_limits<double>::infinity());
    for (unsigned dim = 0; dim < nDim; ++dim) {
        for (unsigned i = 0; i < nDim; ++i) {
            mins[dim] = std::min( mins[dim], front[ processed[i] ].fitness[dim] );
        }
    }
    
    std::vector<double> range(nDim);
    for (unsigned dim = 0; dim < nDim; ++dim) {
        range[dim] = front[ processed[dim] ].fitness[dim] - mins[dim];
    }

    // calculate distances
    std::vector<double> distances(front.size(), std::numeric_limits<double>::infinity());
    
    unsigned selected = 0;
    // select based on maximum distance to nearest processed point
    for (unsigned i = 0; i < front.size(); ++i) {
        
        for (unsigned k = 0; k < processed.size(); ++k) {
            
            if (i==processed[k]) {
                distances[i] = -1.0;
                continue;
            }

            double d = distance( front[i].fitness, front[ processed[k] ].fitness, range );
            
            if (d < distances[i]) {
                distances[i] = d;
            }
            
        }
        
        if (distances[i] > distances[selected]) {
            selected = i;
        }

    }

    while (processed.size() < front.size()) {
       
        // set worth
        typename EOT::Fitness f = parents[ front[selected].index ]->fitness();
        f.setWorth(rank--);
        parents[ front[selected].index ]->fitness(f);
        distances[selected] = -1;

        processed.push_back(selected);

        selected = 0;

        for (unsigned i = 0; i < front.size(); ++i) {
            if (distances[i] < 0) continue;

            double d = distance(front[i].fitness, front[processed.back()].fitness, range);
            
            if (d < distances[i]) {
                distances[i] = d;
            }

            if (distances[i] > distances[selected]) {
                selected = i;
            }
        }

    }
    
    return rank;
  }
};

#endif
