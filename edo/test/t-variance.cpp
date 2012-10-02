
#include <iostream>
#include <vector>

#include <eo>
#include <es.h>
#include <edo>

int main()
{
    typedef eoReal<eoMinimizingFitness> Vec;

    eoPop<Vec> pop;
    for( unsigned int i=1; i<7; ++i) {
        Vec indiv(1,i);
        pop.push_back( indiv );
        std::clog << indiv << " ";
    }
    std::clog << std::endl;

    edoEstimatorNormalMono<Vec> estimator;

    edoNormalMono<Vec> distrib = estimator(pop);

    Vec ex_mean(1,3.5);
    Vec ex_var(1,17.5/6);
    Vec es_mean = distrib.mean();
    Vec es_var = distrib.variance();

    std::cout << "expected  mean=" << ex_mean << " variance=" << ex_var << std::endl;
    std::cout << "estimated mean=" << es_mean << " variance=" << es_var << std::endl;

    for( unsigned int i=0; i<ex_mean.size(); ++i ) {
        assert( es_mean[i] == ex_mean[i] );
        assert(  es_var[i] == ex_var[i] );
    }
}

