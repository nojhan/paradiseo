#ifndef eoNSGA_IIa_Replacement_h
#define eoNSGA_IIa_Replacement_h

#include <moo/eoFrontSorter.h>
#include <eoReplacement.h>

/** @brief Fast Elitist Non-Dominant Sorting Genetic Algorithm

  Adapted from Deb, Agrawal, Pratab and Meyarivan: A Fast Elitist
  Non-Dominant Sorting Genetic Algorithm for MultiObjective
  Optimization: NSGA-II KanGAL Report No. 200001

  Note that this class does not do the sorting per se, but the sorting
  of it worth_std::vector will give the right order

*/
template <class EOT>
class eoNSGA_IIa_Replacement : public eoReplacement<EOT>
{
  public:
  
  void operator()(eoPop<EOT>& parents, eoPop<EOT>& offspring) {
        unsigned origSize = parents.size();

        std::copy(offspring.begin(), offspring.end(), std::back_inserter(parents));

        typename eoFrontSorter<EOT>::front_t front = sorter(parents);
        
        unsigned rank = parents.size();
        for (unsigned i = 0; i < front.size(); ++i) {
            rank = assign_worths(front[i], rank, parents); 
        }
        
        // sort on worth (assuming eoMOFitness) 
        std::sort(parents.begin(), parents.end());
        
        // truncate
        parents.resize(origSize);
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
  
  unsigned assign_worths(const std::vector<detail::FitnessInfo>& front, unsigned rank, eoPop<EOT>& parents) {
    
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
        typename EOT::Fitness f = parents[ front[ processed[i] ].index].fitness();
        f.setWorth(rank);
        parents[ front[ processed[i] ].index ].fitness(f);   
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
        typename EOT::Fitness f = parents[ front[selected].index ].fitness();
        f.setWorth(rank--);
        parents[ front[selected].index ].fitness(f);
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
