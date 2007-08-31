#ifndef __EONDSorting_I_h
#define __EONDSorting_I_h

#include <moo/eoFrontSorter.h>

/**
  The original Non Dominated Sorting algorithm from Srinivas and Deb
*/
template <class EOT>
class eoNSGA_I_Replacement : public eoReplacement<EOT>
{
public :
  eoNSGA_I_Replacement(double _nicheSize) : nicheSize(_nicheSize) {}

  void operator()(eoPop<EOT>& parents, eoPop<EOT>& offspring) {
        
        unsigned origSize = parents.size();

        std::copy(offspring.begin(), offspring.end(), std::back_inserter(parents));

        typename eoFrontSorter<EOT>::front_t front = sorter(parents);

        for (unsigned i = 0; i < front.size(); ++i) {
            assign_worths(front[i], front.size() - i, parents); 
        }
        
        // sort on worth (assuming eoMOFitness) 
        std::sort(parents.begin(), parents.end());
        
        // truncate
        parents.resize(origSize);
  }

  private:
  void assign_worths(const std::vector<detail::FitnessInfo>& front, unsigned rank, eoPop<EOT>& parents) {

        for (unsigned i = 0; i < front.size(); ++i)
        { // calculate whether the other points lie within the nice
          double niche_count = 0;

          for (unsigned j = 0; j < front.size(); ++j)
          {
            if (i == j)
              continue;

            double dist = 0.0;

            for (unsigned k = 0; k < front[j].fitness.size(); ++k)
            {
              double d = front[i].fitness[k] - front[j].fitness[k];
              dist += d*d;
            }

            if (dist < nicheSize)
            {
              niche_count += 1.0 - pow(dist / nicheSize,2.);
            }
          }
        
          unsigned idx = front[i].index;
          typename EOT::Fitness f = parents[idx].fitness();
          f.setWorth(rank + niche_count);
          parents[ idx ].fitness(f);
        }
  }

  private :

  double nicheSize;
  eoFrontSorter<EOT> sorter;
};

#endif


