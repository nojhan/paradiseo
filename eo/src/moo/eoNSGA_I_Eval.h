#ifndef __eoNSGA_I_Eval_h_
#define __eoNSGA_I_Eval_h_

#include <moo/eoFrontSorter.h>
#include <moo/eoMOEval.h>

/**
  The original Non Dominated Sorting algorithm from Srinivas and Deb
*/
template <class EOT>
class eoNSGA_I_Eval : public eoMOEval<EOT>
{
public :
  eoNSGA_I_Eval(double nicheWidth, eoEvalFunc<EOT>& eval)    : eoMOEval<EOT>(eval), nicheSize(nicheWidth) {}
  eoNSGA_I_Eval(double nicheWidth, eoPopEvalFunc<EOT>& eval) : eoMOEval<EOT>(eval), nicheSize(nicheWidth) {}

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

  private:
  void assign_worths(const std::vector<detail::FitnessInfo>& front, unsigned rank, std::vector<EOT*>& parents) {

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
          typename EOT::Fitness f = parents[idx]->fitness();
          f.setWorth(rank + niche_count);
          parents[ idx ]->fitness(f);
        }
  }

  private :

  double nicheSize;
  eoFrontSorter<EOT> sorter;
};

#endif


