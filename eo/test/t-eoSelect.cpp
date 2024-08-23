// to avoid long name warnings
#ifdef _MSC_VER
#pragma warning(disable:4786)
#endif

#include <cstring>
#include <stdexcept>

#include <eo>

//-----------------------------------------------------------------------------

struct Dummy : public EO<double>
{
    typedef double Type;
  void printOn(std::ostream & _os) const
  {
      _os << " - ";
      EO<double>::printOn(_os);
  }
};

bool operator==(const Dummy & _d1, const Dummy & _d2)
{
  return _d1.fitness() == _d2.fitness();
}

struct eoDummyPop : public eoPop<Dummy>
{
public :
    eoDummyPop(int s=0) { resize(s); }
};

// helper - DOES NOT WORK if different individuals have same fitness!!!
template <class EOT>
unsigned isInPop(EOT & _indi, eoPop<EOT> & _pop)
{
  for (unsigned i=0; i<_pop.size(); i++)
    if (_pop[i] == _indi)
      return i;
  return _pop.size();
}

unsigned int pSize;		// global variable, bouh!
std::string fitnessType;		// yes, a global variable :-)
eoDummyPop parentsOrg;

template <class EOT>
void testSelectMany(eoSelect<EOT> & _select, std::string _name)
{
    unsigned i;
  std::cout << "\n\n" << fitnessType + _name << std::endl;
  std::cout << "===============\n";

    eoDummyPop parents(parentsOrg);
    eoDummyPop offspring(0);

    // do the selection
    _select(parents, offspring);

    // compute stats
    std::vector<unsigned> nb(parents.size(), 0);
    for (i=0; i<offspring.size();  i++)
      {
	unsigned trouve = isInPop<Dummy>(offspring[i], parents);
	if (trouve == parents.size()) // pas trouve
	  throw eoException("Pas trouve ds parents");
	nb[trouve]++;
       }
    // dump to file so you can plot using gnuplot - dir name is hardcoded!
    std::string fName = "ResSelect/" + fitnessType + _name + ".select";
    std::ofstream os(fName.c_str());
    for (i=0; i<parents.size();  i++)
      {
	std::cout << i << " -> " << ( (double)nb[i])/offspring.size() << std::endl;
	os << i << " " << ( (double)nb[i])/offspring.size() << std::endl;
      }

}

template <class EOT>
void testSelectOne(eoSelectOne<EOT> & _select, eoHowMany & _offspringRate,
		   eoHowMany & _fertileRate, std::string _name)
{
  eoTruncatedSelectOne<EOT> truncSelect(_select, _fertileRate);
  eoSelectMany<EOT> percSelect(truncSelect, _offspringRate);
  testSelectMany<EOT>(percSelect, _name);
}


//-----------------------------------------------------------------------------

int the_main(int argc, char **argv)
{
  eoParser parser(argc, argv);
  eoValueParam<unsigned> parentSizeParam = parser.createParam(unsigned(10), "parentSize", "Parent size",'P');
    pSize = parentSizeParam.value(); // global variable

//   eoValueParam<double> offsrpringRateParam = parser.createParam<double>(1.0, "offsrpringRate", "Offsrpring rate",'O');
//     double oRate = offsrpringRateParam.value();
  eoValueParam<eoHowMany> offsrpringRateParam = parser.createParam(eoHowMany(1.0), "offsrpringRate", "Offsrpring rate (% or absolute)",'O');
    eoHowMany oRate = offsrpringRateParam.value();

  eoValueParam<eoHowMany> fertileRateParam = parser.createParam(eoHowMany(1.0), "fertileRate", "Fertility rate (% or absolute)",'F');
    eoHowMany fRate = fertileRateParam.value();

eoValueParam<unsigned> tournamentSizeParam = parser.createParam(unsigned(2), "tournamentSize", "Deterministic tournament size",'T');
    unsigned int tSize = tournamentSizeParam.value();

  eoValueParam<double> tournamentRateParam = parser.createParam(1.0, "tournamentRate", "Stochastic tournament rate",'t');
    double tRate = tournamentRateParam.value();

  eoValueParam<double> rankingPressureParam = parser.createParam(2.0, "rankingPressure", "Selective pressure for the ranking selection",'p');
    double rankingPressure = rankingPressureParam.value();

  eoValueParam<double> rankingExponentParam = parser.createParam(1.0, "rankingExponent", "Exponent for the ranking selection",'e');
    double rankingExponent = rankingExponentParam.value();

  eoValueParam<std::string> fitTypeParam = parser.createParam(std::string("linear"), "fitType", "Type of fitness (linear, exp, log, super",'f');
    fitnessType = fitTypeParam.value();

    if (parser.userNeedsHelp())
      {
	parser.printHelp(std::cout);
	exit(0);
      }

    // hard-coded directory name ...
    const int ignored = system("mkdir ResSelect");
    std::cout << "Testing the Selections\nParents size = " << pSize
	 << ", offspring rate = " << oRate;
    std::cout << " and putting rsulting files in dir ResSelect" << std::endl;

    // initialize parent population
    parentsOrg.resize(pSize);
    if (fitnessType == std::string("linear"))
      for (unsigned i=0; i<pSize; i++)
	parentsOrg[i].fitness(i);
    else if (fitnessType == std::string("exp"))
      for (unsigned i=0; i<pSize; i++)
	parentsOrg[i].fitness(exp((double)i));
    else if (fitnessType == std::string("log"))
      for (unsigned i=0; i<pSize; i++)
	parentsOrg[i].fitness(log(i+1.));
    else if (fitnessType == std::string("super"))
      {
	for (unsigned i=0; i<pSize-1; i++)
	  parentsOrg[i].fitness(i);
	parentsOrg[pSize-1].fitness(10*pSize);
      }
    else
      throw eoInvalidFitnessError("Invalid fitness Type"+fitnessType);

    std::cout << "Initial parents (odd)\n" << parentsOrg << std::endl;

  // random seed
    eoValueParam<uint32_t>& seedParam = parser.createParam(uint32_t(0), "seed",
							   "Random number seed", 'S');
    if (seedParam.value() == 0)
	seedParam.value() = time(0);
    rng.reseed(seedParam.value());

    char fileName[1024];

// the selection procedures under test
    //    eoDetSelect<Dummy> detSelect(oRate);
    //    testSelectMany(detSelect, "detSelect");

    // Roulette
     eoProportionalSelect<Dummy> propSelect;
     testSelectOne<Dummy>(propSelect, oRate, fRate, "PropSelect");

    // Linear ranking using the perf2Worth construct
    eoRankingSelect<Dummy> newRankingSelect(rankingPressure);
    sprintf(fileName,"LinRank_%g",rankingPressure);
    testSelectOne<Dummy>(newRankingSelect, oRate, fRate, fileName);

    // Exponential ranking using the perf2Worth construct
    std::cout << "rankingExponent " << rankingExponent << std::endl;
    eoRankingSelect<Dummy> expRankingSelect(rankingPressure,rankingExponent);
    sprintf(fileName,"ExpRank_%g_%g",rankingPressure, rankingExponent);
    testSelectOne<Dummy>(expRankingSelect, oRate, fRate, fileName);

    // Det tournament
    eoDetTournamentSelect<Dummy> detTourSelect(tSize);
    sprintf(fileName,"DetTour_%d",tSize);
    testSelectOne<Dummy>(detTourSelect, oRate, fRate, fileName);

    // Stoch tournament
    eoStochTournamentSelect<Dummy> stochTourSelect(tRate);
    sprintf(fileName,"StochTour_%g",tRate);
    testSelectOne<Dummy>(stochTourSelect, oRate, fRate, fileName);

    // Fitness scaling
    eoFitnessScalingSelect<Dummy> newFitScaleSelect(rankingPressure);
    sprintf(fileName,"LinFitScale_%g",rankingPressure);
    testSelectOne<Dummy>(newFitScaleSelect, oRate, fRate, fileName);

    // Sequential selections
    eoSequentialSelect<Dummy> seqSel(false);
    strcpy(fileName,"Sequential");
    testSelectOne<Dummy>(seqSel, oRate, fRate, fileName);

    eoEliteSequentialSelect<Dummy> eliteSeqSel;
    strcpy(fileName,"EliteSequential");
    testSelectOne<Dummy>(eliteSeqSel, oRate, fRate, fileName);

    return 1;
}

int main(int argc, char **argv)
{
    try
    {
	the_main(argc, argv);
    }
    catch(std::exception& e)
    {
	std::cout << "Exception: " << e.what() << std::endl;
	return 1;
    }
}
