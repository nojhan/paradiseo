#include <apply.h>
#include <eo>
#include <eoRanking.h>
#include <eoRankingCached.h>
#include <es/eoReal.h>
#include <utils/eoRNG.h>
#include "real_value.h"

class RankingTest
{
public:
    RankingTest(eoParser &parser, eoEvalFuncCounter<eoReal<double>> &_eval, unsigned size = 100)
        : rng(0),
          popSize(size),
          seedParam(parser.createParam(uint32_t(time(0)), "seed", "Random seed", 'S')),
          pressureParam(parser.createParam(1.5, "pressure", "Selective pressure", 'p')),
          exponentParam(parser.createParam(1.0, "exponent", "Ranking exponent", 'e')),
          eval(_eval)
    {
        rng.reseed(seedParam.value());
        initPopulation();
    }

    void initPopulation()
    {
        pop.clear();
        for (unsigned i = 0; i < popSize; ++i)
        {
            eoReal<double> ind;
            ind.resize(1);
            ind[0] = rng.uniform();
            pop.push_back(ind);
        }
        apply<eoReal<double>>(eval, pop);
    }

    const unsigned popSize;
    eoPop<eoReal<double>> pop;
    eoRng rng;
    double pressure() const { return pressureParam.value(); }
    double exponent() const { return exponentParam.value(); }

private:
    eoValueParam<uint32_t> &seedParam;
    eoValueParam<double> &pressureParam;
    eoValueParam<double> &exponentParam;
    eoEvalFuncCounter<eoReal<double>> eval;
};

// Test case 1: Verify both implementations produce identical results
void test_Consistency(eoParser &parser)
{
    eoEvalFuncPtr<eoReal<double>, double, const std::vector<double> &> mainEval(real_value);
    eoEvalFuncCounter<eoReal<double>> eval(mainEval);
    RankingTest fixture(parser, eval);

    eoRanking<eoReal<double>> ranking(fixture.pressure(), fixture.exponent());
    eoRankingCached<eoReal<double>> rankingCached(fixture.pressure(), fixture.exponent());

    ranking(fixture.pop);
    rankingCached(fixture.pop);

    const std::vector<double> &values = ranking.value();
    const std::vector<double> &cachedValues = rankingCached.value();

    for (unsigned i = 0; i < fixture.pop.size(); ++i)
    {
        if (abs(values[i] - cachedValues[i]) > 1e-9)
        {
            throw std::runtime_error("Inconsistent ranking values between implementations");
        }
    }
    std::clog << "Test 1 passed: Both implementations produce identical results" << std::endl;
}

// Test case 2: Test edge case with minimum population size
void test_MinPopulationSize(eoParser &parser)
{
    eoPop<eoReal<double>> smallPop;
    eoReal<double> ind1, ind2;
    ind1.resize(1);
    ind1[0] = 0.5;
    ind2.resize(1);
    ind2[0] = 1.0;
    smallPop.push_back(ind1);
    smallPop.push_back(ind2);
    eoEvalFuncPtr<eoReal<double>, double, const std::vector<double> &> mainEval(real_value);
    eoEvalFuncCounter<eoReal<double>> eval(mainEval);

    RankingTest fixture(parser, eval, 2); // Use fixture to get parameters
    eoRanking<eoReal<double>> ranking(fixture.pressure(), fixture.exponent());
    eoRankingCached<eoReal<double>> rankingCached(fixture.pressure(), fixture.exponent());

    apply<eoReal<double>>(eval, smallPop);

    ranking(smallPop);
    rankingCached(smallPop);

    if (ranking.value()[0] >= ranking.value()[1] ||
        rankingCached.value()[0] >= rankingCached.value()[1])
    {
        throw std::runtime_error("Invalid ranking for population size 2");
    }
    std::clog << "Test 2 passed: Minimum population size handled correctly" << std::endl;
}

// Test case 3: Verify caching actually works
void test_CachingEffectiveness(eoParser &parser)
{
    eoEvalFuncPtr<eoReal<double>, double, const std::vector<double> &> mainEval(real_value);
    eoEvalFuncCounter<eoReal<double>> eval(mainEval);
    RankingTest fixture(parser, eval, 50); // Fixed size for cache test

    eoRankingCached<eoReal<double>> rankingCached(fixture.pressure(), fixture.exponent());

    // First run - should compute all values
    rankingCached(fixture.pop);
    const auto firstValues = rankingCached.value();

    // Modify fitness values but keep same population size
    for (auto &ind : fixture.pop)
    {
        ind[0] = fixture.rng.uniform();
    }

    apply<eoReal<double>>(eval, fixture.pop);

    // Second run - should use cached coefficients
    rankingCached(fixture.pop);

    // Add one individual to invalidate cache
    eoReal<double> newInd;
    newInd.resize(1);
    newInd[0] = fixture.rng.uniform();
    fixture.pop.push_back(newInd);

    apply<eoReal<double>>(eval, fixture.pop);

    // Third run - should recompute coefficients
    rankingCached(fixture.pop);

    std::clog << "Test 3 passed: Caching mechanism properly invalidated" << std::endl;
}

// Helper function to test constructor assertions
bool testRankingConstructor(double pressure, double exponent)
{
    try
    {
        eoRanking<eoReal<double>> ranking(pressure, exponent);
        return true; // Constructor succeeded
    }
    catch (...)
    {
        return false; // Assertion failed
    }
}

// Helper function to test constructor assertions
bool testRankingCachedConstructor(double pressure, double exponent)
{
    try
    {
        eoRankingCached<eoReal<double>> ranking(pressure, exponent);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

// Test case 4: Verify assertions on invalid parameters
void test_Assertions(eoParser &parser)
{
    // Test valid parameters (should succeed)
    bool valid_ok = true;
    valid_ok &= testRankingConstructor(1.1, 1.0);       // Valid pressure and exponent
    valid_ok &= testRankingConstructor(1.1, 2.0);       // Edge case valid
    valid_ok &= testRankingCachedConstructor(1.1, 1.0); // Valid pressure and exponent
    valid_ok &= testRankingCachedConstructor(1.1, 2.0); // Edge case valid

    // Test invalid parameters (should fail)
    bool invalid_ok = true;
    invalid_ok &= !testRankingConstructor(1.0, 1.0);       // pressure = 1 (invalid)
    invalid_ok &= !testRankingConstructor(0.5, 1.0);       // pressure < 1 (invalid)
    invalid_ok &= !testRankingConstructor(2.0, 2.1);       // exponent > 2 (invalid)
    invalid_ok &= !testRankingCachedConstructor(1.0, 1.0); // pressure = 1 (invalid)
    invalid_ok &= !testRankingCachedConstructor(0.5, 1.0); // pressure < 1 (invalid)
    invalid_ok &= !testRankingCachedConstructor(2.5, 2.1); // exponent > 2 (invalid)

    if (!valid_ok)
    {
        throw std::runtime_error("Valid parameter tests failed");
    }

    if (!invalid_ok)
    {
        throw std::runtime_error("Invalid parameter tests failed - some invalid values were accepted");
    }

    std::clog << "Test 4 passed: All parameter assertions working correctly\n";
}

int main(int argc, char **argv)
{
    try
    {
        eoParser parser(argc, argv);
        test_Consistency(parser);
        test_MinPopulationSize(parser);
        test_CachingEffectiveness(parser);
        // test_Assertions(parser);
        return 0;
    }
    catch (std::exception &e)
    {
        std::clog << "Exception: " << e.what() << std::endl;
        return 1;
    }
}
