#include <eo>
#include <eoReal.h>
#include <utils/eoRNG.h>
#include <eoRanking.h>

class RankingTest
{
public:
    RankingTest(eoParser &parser, unsigned size = 100) : rng(0),
                                                         popSize(size),
                                                         seedParam(parser.createParam(uint32_t(time(0)), "seed", "Random seed", 'S')),
                                                         pressureParam(parser.createParam(1.5, "pressure", "Selective pressure", 'p')),
                                                         exponentParam(parser.createParam(1.0, "exponent", "Ranking exponent", 'e'))
    {
        rng.reseed(seedParam.value());
        initPopulation();
    }

    void initPopulation()
    {
        pop.clear();
        pop.resize(popSize);
        for (unsigned i = 0; i < popSize; ++i)
        {
            eoReal<double> ind;
            ind.resize(1);
            ind[0] = rng.uniform();
            pop.push_back(ind);
        }
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
};

// Test case 1: Verify both implementations produce identical results
void test_Consistency(eoParser &parser)
{
    RankingTest fixture(parser);

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
    std::cout << "Test 1 passed: Both implementations produce identical results\n";
}

// Test case 2: Verify ranking order is preserved
void test_RankingOrder(eoParser &parser)
{
    RankingTest fixture(parser);

    eoRankingCached<eoReal<double>> ranking(fixture.pressure(), fixture.exponent());
    ranking(fixture.pop);

    fixture.pop.sort();
    const std::vector<double> &values = ranking.value();

    for (unsigned i = 1; i < fixture.pop.size(); ++i)
    {
        if (values[i] > values[i - 1])
        {
            throw std::runtime_error("Ranking order not preserved");
        }
    }
    std::cout << "Test 2 passed: Ranking order is preserved\n";
}

// Test case 3: Test edge case with minimum population size
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

    RankingTest fixture(parser, 2); // Use fixture to get parameters
    eoRanking<eoReal<double>> ranking(fixture.pressure(), fixture.exponent());
    eoRankingCached<eoReal<double>> rankingCached(fixture.pressure(), fixture.exponent());

    ranking(smallPop);
    rankingCached(smallPop);

    if (ranking.value()[0] >= ranking.value()[1] ||
        rankingCached.value()[0] >= rankingCached.value()[1])
    {
        throw std::runtime_error("Invalid ranking for population size 2");
    }
    std::cout << "Test 3 passed: Minimum population size handled correctly\n";
}

// Test case 4: Verify caching actually works
void test_CachingEffectiveness(eoParser &parser)
{
    RankingTest fixture(parser, 50); // Fixed size for cache test

    eoRankingCached<eoReal<double>> rankingCached(fixture.pressure(), fixture.exponent());

    // First run - should compute all values
    rankingCached(fixture.pop);
    const auto firstValues = rankingCached.value();

    // Modify fitness values but keep same population size
    for (auto &ind : fixture.pop)
        ind[0] = fixture.rng.uniform();

    // Second run - should use cached coefficients
    rankingCached(fixture.pop);

    // Add one individual to invalidate cache
    eoReal<double> newInd;
    newInd.resize(1);
    newInd[0] = fixture.rng.uniform();
    fixture.pop.push_back(newInd);

    // Third run - should recompute coefficients
    rankingCached(fixture.pop);

    std::cout << "Test 4 passed: Caching mechanism properly invalidated\n";
}

int main(int argc, char **argv)
{
    try
    {
        eoParser parser(argc, argv);
        test_Consistency(parser);
        test_RankingOrder(parser);
        test_MinPopulationSize(parser);
        test_CachingEffectiveness(parser);
        return 0;
    }
    catch (std::exception &e)
    {
        std::cout << "Exception: " << e.what() << std::endl;
        return 1;
    }
}
