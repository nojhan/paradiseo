//-----------------------------------------------------------------------------

// to avoid long name warnings
#ifdef _MSC_VER
#pragma warning(disable:4786)
#endif

#include <stdexcept>  // runtime_error 

//-----------------------------------------------------------------------------
// tt.cpp: 
//
//-----------------------------------------------------------------------------


// general
#include <eo>
#include <eoFitnessScalingSelect.h>
#include <eoSelectFromWorth.h>
#include <eoLinearFitScaling.h>
//-----------------------------------------------------------------------------

struct Dummy : public EO<double>
{
    typedef double Type;
  void printOn(ostream & _os) const
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

template <class EOT>
void testSelectMany(eoSelect<EOT> & _select, string _name)
{
  cout << "\n\n" << _name << endl;
  cout << "===============\n"; 

    eoDummyPop parents(pSize);
    eoDummyPop offspring(0);
    unsigned i;
    // initialize parents
    for (i=0; i<pSize; i++)
      //      parents[i].fitness(log(i+1));
      //      parents[i].fitness(exp(i));
      parents[i].fitness(i);
    cout << "Initial parents (odd)\n" << parents << endl;

    // do the selection
    _select(parents, offspring);

    // compute stats
    vector<unsigned> nb(parents.size(), 0);
    for (i=0; i<offspring.size();  i++)
      {
	unsigned trouve = isInPop<Dummy>(offspring[i], parents);
	if (trouve == parents.size()) // pas trouve
	  throw runtime_error("Pas trouve ds parents");
	nb[trouve]++;
       }
    // dump to file so you can plot using gnuplot
    string fName = _name + ".prop";
    ofstream os(fName.c_str());
    for (i=0; i<parents.size();  i++)
      {
	cout << i << " -> " << ( (double)nb[i])/offspring.size() << endl;
	os << i << " " << ( (double)nb[i])/offspring.size() << endl;
      }

}

template <class EOT>
void testSelectOne(eoSelectOne<EOT> & _select, eoHowMany & _hm, string _name)
{
  eoSelectMany<EOT> percSelect(_select, _hm);
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

eoValueParam<unsigned> tournamentSizeParam = parser.createParam(unsigned(2), "tournamentSize", "Deterministic tournament size",'T');
    unsigned int tSize = tournamentSizeParam.value();

  eoValueParam<double> tournamentRateParam = parser.createParam(0.75, "tournamentRate", "Stochastic tournament rate",'R');
    double tRate = tournamentRateParam.value();

  eoValueParam<double> rankingPressureParam = parser.createParam(1.75, "rankingPressure", "Selective pressure for the ranking selection",'p');
    double rankingPressure = rankingPressureParam.value();

    if (parser.userNeedsHelp())
      {
        parser.printHelp(cout);
        exit(0);
      }

    cout << "Testing the Selections\nParents size = " << pSize 
	 << ", offspring rate = " << oRate << endl;

    rng.reseed(42);


// the selection procedures under test
    //    eoDetSelect<Dummy> detSelect(oRate);
    //    testSelectMany(detSelect, "detSelect");

    // Roulette
     eoProportionalSelect<Dummy> propSelect;
     testSelectOne<Dummy>(propSelect, oRate, "propSelect");

    // Ranking
     eoRankingSelect<Dummy> rankSelect(rankingPressure);
     testSelectOne<Dummy>(rankSelect, oRate, "rankSelect");

    // New ranking using the perf2Worth construct
      cout << "Avant appel a LinearRanking()" << endl;
    eoRankingSelect<Dummy> newRankingSelect(rankingPressure); // pressure 2 by default
    testSelectOne<Dummy>(newRankingSelect, oRate, "newRankSelect");

    // New ranking using the perf2Worth construct
      cout << "Avant appel a exponentialRanking()" << endl;
    eoRankingSelect<Dummy> expRankingSelect(rankingPressure,2);
    testSelectOne<Dummy>(expRankingSelect, oRate, "expRankingSelect");

    // Det tournament
    eoDetTournamentSelect<Dummy> detTourSelect(tSize);
    testSelectOne<Dummy>(detTourSelect, oRate, "detTourSelect");

    // Stoch tournament
    eoStochTournamentSelect<Dummy> stochTourSelect(tRate);
    testSelectOne<Dummy>(stochTourSelect, oRate, "stochTourSelect");

    exit(0);

    // Fitness scaling
//     eoFitnessScalingSelect<Dummy> fitScaleSelect(rankingPressure);
//     testSelectOne<Dummy>(fitScaleSelect, oRate, "fitScaleSelect");

    // NEW Fitness scaling
    eoFitnessScalingSelect<Dummy> newFitScaleSelect(rankingPressure);
    testSelectOne<Dummy>(newFitScaleSelect, oRate, "NewFitScaleSelect");

    return 1;
}

int main(int argc, char **argv)
{
    try
    {
        the_main(argc, argv);
    }
    catch(exception& e)
    {
        cout << "Exception: " << e.what() << endl;
        return 1;
    }
}
