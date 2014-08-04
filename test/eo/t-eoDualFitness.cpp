#include <utility>

#include <paradiseo/eo.h>
#include <paradiseo/eo/es.h>
#include <paradiseo/eo/utils/eoStat.h>

typedef eoVector<eoDualFitness<double,eoMinimizingFitness>,double> DualVector;

template<class EOT>
class DualSphere : public eoEvalFunc<EOT>
{
public:
    virtual void operator()( EOT & x )
    {
	if( x.invalid() ) { return; }

	double sum = 0;
	int sign = 1;
	for( unsigned int i=0, s=x.size(); i<s; ++i ) {
	    sum += x[i] * x[i];
	    sign *= x[i]<0 ? -1 : 1;
	}

	x.fitness( std::make_pair( sum, sign>0 ? true : false ) );
    }
};


double test( eoPop<DualVector>& pop, double target_value )
{
    DualSphere<DualVector> eval;

    eoPopLoopEval<DualVector> pop_eval(eval);

    pop_eval(pop,pop);

    eoInterquartileRangeStat<DualVector> iqr_stat( std::make_pair(0.0,false), "IQR" );

    iqr_stat( pop );

    std::cout << iqr_stat.longName() << "=" << iqr_stat.value() << " should be " << target_value << std::endl;

    return iqr_stat.value().value();
}


int main()
{
    eoPop<DualVector> pop;

    // fixed test
    DualVector sol1(2,-1);
    DualVector sol2(2,-1);
    DualVector sol3(2,1);
    DualVector sol4(2,1);
    pop.push_back( sol1 );
    pop.push_back( sol2 );
    pop.push_back( sol3 );
    pop.push_back( sol4 );
    // on the sphere function everyone has the same fitness of 1
    if( test(pop, 0) != 0 ) {
	exit(1);
    }

    pop.erase(pop.begin(),pop.end());

    // fixed test
    sol1 = DualVector(2,0);
    sol2 = DualVector(2,0);
    sol3 = DualVector(2,1);
    sol4 = DualVector(2,1);
    pop.push_back( sol1 );
    pop.push_back( sol2 );
    pop.push_back( sol3 );
    pop.push_back( sol4 );
    if( test(pop, 1) != 1 ) {
	exit(1);
    }

    // test on a random normal distribution
    eoNormalGenerator<double> normal(1,rng);
    eoInitFixedLength<DualVector> init_N(2, normal);
    pop = eoPop<DualVector>( 1000000, init_N );
    double iqr = test(pop, 1.09);
    if( iqr < 1.08 || iqr > 1.11 ) {
	exit(1);
    }
}
