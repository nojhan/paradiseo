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
#include <eoDetSelect.h>
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


struct eoDummyPop : public eoPop<Dummy>
{
public :
    eoDummyPop(int s=0) { resize(s); }
};

//-----------------------------------------------------------------------------

int the_main(int argc, char **argv)
{ 
  eoParser parser(argc, argv);
  eoValueParam<unsigned int> parentSizeParam = parser.createParam<unsigned int>(10, "parentSize", "Parent size",'P');
    unsigned int pSize = parentSizeParam.value();

  eoValueParam<double> offsrpringRateParam = parser.createParam<double>(1.0, "offsrpringRate", "Offsrpring rate",'O');
    double oRate = offsrpringRateParam.value();

  eoValueParam<bool> interpretAsRateParam = parser.createParam<bool>(true, "interpretAsRate", "interpret rate as Rate (False = as Number)",'b');
    bool interpretAsRate = interpretAsRateParam.value();

eoValueParam<unsigned int> tournamentSizeParam = parser.createParam<unsigned int>(2, "tournamentSize", "Deterministic tournament size",'T');
    unsigned int tSize = tournamentSizeParam.value();

  eoValueParam<double> tournamentRateParam = parser.createParam<double>(0.75, "tournamentRate", "Stochastic tournament rate",'R');
    double tRate = tournamentRateParam.value();

    if (parser.userNeedsHelp())
      {
        parser.printHelp(cout);
        exit(1);
      }

    unsigned i;

    cout << "Testing the Selections\nParents size = " << pSize 
	 << ", offspring rate = " << oRate << 
      "interpreted as " << (interpretAsRate ? "Rate" : "Number") << endl;

    rng.reseed(42);


    eoDummyPop parents(pSize);
    eoDummyPop offspring(0);

    // initialize so we can recognize them later!
    for (i=0; i<pSize; i++)
      parents[i].fitness(i);

cout << "Initial parents (odd)\n" << parents << endl;

// the selection procedures under test
    eoDetSelect<Dummy> detSelect(oRate, interpretAsRate);

    // here we go
    // Deterministic
    cout << "eoDetSelect\n";
    cout << "===========\n";
    detSelect(parents, offspring);
cout << "Selected offsprings (origonally all even\n" << offspring << endl;

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
    }

    return 1;
}
