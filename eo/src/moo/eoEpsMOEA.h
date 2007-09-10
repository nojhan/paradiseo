#ifndef eoEpsMOEA_h
#define eoEpsMOEA_h

#include <eoAlgo.h>
#include <moo/eoEpsilonArchive.h>
#include <utils/eoStat.h>


template <class EOT>
class eoEpsMOEA : public eoAlgo<EOT> {
 
    public:
  
    eoEpsMOEA(
         eoContinue<EOT>& _continuator,
         eoEvalFunc<EOT>& _eval,
         eoGenOp<EOT>& _op,
         const std::vector<double>& eps,
         unsigned max_archive_size
     ) : continuator(_continuator),
	 eval (_eval),
	 loopEval(_eval),
	 popEval(loopEval),
         op(_op),
         archive(eps, max_archive_size)
         {
         }

   
    void operator()(eoPop<EOT>& pop) {
        
        eoPop<EOT> offspring;
        popEval(offspring, pop);
        for (unsigned i = 0; i < pop.size(); ++i) pop[i].fitnessReference().setWorth(1.0);

        do {
            unsigned nProcessed = 0;

            while (nProcessed < pop.size()) {
                offspring.clear();

                epsPopulator populator(archive, pop, offspring); 
                op(populator);
                
                nProcessed += offspring.size();
                popEval(pop, offspring);

                for (unsigned i = 0; i < offspring.size(); ++i) {
                    offspring[i].fitnessReference().setWorth(1.0);
                    archive(offspring[i]);
                    update_pop(pop, offspring[i]);
                }
            }

            // quite expensive to copy the entire archive time and time again, but this will make it work more seamlessly with the rest of EO
            offspring.clear();
            archive.appendTo(offspring);

        } while (continuator(offspring)); // check archive
        
        // return archive
        pop.clear();
        archive.appendTo(pop);

    }
    
    private :
    void update_pop(eoPop<EOT>& pop, const EOT& offspring) {
        std::vector<unsigned> dominated;

        for (unsigned i = 0; i < pop.size(); ++i) {
            int dom = offspring.fitness().check_dominance(pop[i].fitness());
            switch (dom) {
                case 1 : // indy dominates this 
                    dominated.push_back(i);
                    break;
                case -1 : // is dominated, do not insert
                    return;
                case 0: // incomparable
                    break;
            }
        }
        
        if (dominated.size()) {
            pop[ dominated[ rng.random(dominated.size()) ] ] = offspring;
        }

        // non-dominated everywhere, overwrite random one
        pop[ rng.random(pop.size()) ] = offspring;
    }

    class epsPopulator : public eoPopulator<EOT> {
        
        using eoPopulator< EOT >::src;

        eoEpsilonArchive<EOT>& archive;
        bool fromArchive;
        public:
        epsPopulator(eoEpsilonArchive<EOT>& arch, const eoPop<EOT>& pop, eoPop<EOT>& res) : eoPopulator<EOT>(pop, res), archive(arch), fromArchive(true) {}
    
        const EOT& select() {
            fromArchive = !fromArchive;
            
            using std::cout;
            using std::endl;
            
            if (fromArchive && !archive.empty()) {
                return archive.selectRandom();
            }
            
            // tournament selection on population
            const EOT& eo1 = rng.choice(src);
            const EOT& eo2 = rng.choice(src);

            if (eo1.fitness().dominates(eo2.fitness())) return eo1;
            return eo2; // they are drawn at random, so no need to do an extra randomization step
        }

    };


  eoContinue<EOT>&          continuator;

  eoEvalFunc <EOT> &        eval ;
  eoPopLoopEval<EOT>        loopEval;

  eoPopEvalFunc<EOT>&       popEval;
  
  eoGenOp<EOT>& op;

  eoEpsilonArchive<EOT> archive;
    
};

#endif

