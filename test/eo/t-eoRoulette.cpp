#include <paradiseo/eo/eoPop.h>
#include <paradiseo/eo/EO.h>
#include <paradiseo/eo/eoProportionalSelect.h>
#include <paradiseo/eo/eoStochasticUniversalSelect.h>

class TestEO : public EO<double> { public: unsigned index; };

using namespace std;

template <class Select>
int test_select()
{
    vector<double> probs(4);
    probs[0] = 0.1;
    probs[1] = 0.4;
    probs[2] = 0.2;
    probs[3] = 0.3;

    vector<double> counts(4,0.0);

    // setup population
    eoPop<TestEO> pop;
    for (unsigned i = 0; i < probs.size(); ++i)
    {
	pop.push_back( TestEO());
	pop.back().fitness( probs[i] * 2.1232 ); // some number to check scaling
	pop.back().index = i;
    }

    Select select;

    unsigned ndraws = 10000;

    for (unsigned i = 0; i < ndraws; ++i)
    {
	const TestEO& eo = select(pop);

	counts[eo.index]++;
    }

    cout << "Threshold = " << 1./sqrt(double(ndraws)) << endl;

    for (unsigned i = 0; i < 4; ++i)
    {
	cout << counts[i]/ndraws << ' ';

	double c = counts[i]/ndraws;

	if (fabs(c - probs[i]) > 1./sqrt((double)ndraws)) {
	    cout << "ERROR" << endl;
	    return 1;
	}
    }

    cout << endl;
    return 0;
}

int main()
{
    rng.reseed(44);

    if (test_select<eoProportionalSelect<TestEO> >()) return 1;

    return test_select<eoStochasticUniversalSelect<TestEO> >();
}
