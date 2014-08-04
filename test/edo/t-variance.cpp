
#include <iostream>
#include <vector>

#include <paradiseo/eo.h>
#include <paradiseo/edo.h>
#include <paradiseo/eo/es.h> 

typedef eoReal<eoMinimizingFitness> Vec;

void check( Vec ex_mean, Vec ex_var, Vec es_mean, Vec es_var )
{
    std::cout << "expected  mean=" << ex_mean << " variance=" << ex_var << std::endl;
    std::cout << "estimated mean=" << es_mean << " variance=" << es_var << std::endl;

    for( unsigned int i=0; i<ex_mean.size(); ++i ) {
        assert( es_mean[i] == ex_mean[i] );
        assert(  es_var[i] == ex_var[i] );
    }
}


int main()
{

    std::clog << "Variance computation on a simple vector" << std::endl;
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

    check( ex_mean, ex_var, es_mean, es_var );


    // Test with negative values
    std::clog << "Variance computation on negative values" << std::endl;
    pop.clear();

    pop.push_back( Vec(1, 11.0) );
    pop.push_back( Vec(1, -4.0) );
    pop.push_back( Vec(1, 11.0) );
    pop.push_back( Vec(1, 14.0) );
    pop.push_back( Vec(1, -2.0) );

    std::clog << pop << std::endl;

    distrib = estimator(pop);

    ex_mean = Vec(1,6.0);
    ex_var = Vec(1,278.0/5);
    es_mean = distrib.mean();
    es_var = distrib.variance();

    check( ex_mean, ex_var, es_mean, es_var );


    // test single individual
    std::clog << "Variance computation on a pop with a single individual" << std::endl;
    pop.clear();
    pop.push_back( Vec(1, 0.0) );

    distrib = estimator(pop);

    ex_mean = Vec(1,0.0);
    ex_var = Vec(1,0.0);
    es_mean = distrib.mean();
    es_var = distrib.variance();

    check( ex_mean, ex_var, es_mean, es_var );
}

