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
  eoValueParam<unsigned int>& parentSizeParam = parser.createParam<unsigned int>(10, "parentSize", "Parnet size",'P');
    unsigned int pSize = parentSizeParam.value();

  eoValueParam<unsigned int>& offsrpringSizeParam = parser.createParam<unsigned int>(10, "offsrpringSize", "Offsrpring size",'O');
    unsigned int oSize = offsrpringSizeParam.value();

  eoValueParam<unsigned int>& tournamentSizeParam = parser.createParam<unsigned int>(2, "tournamentSize", "Deterministic tournament size",'T');
    unsigned int tSize = tournamentSizeParam.value();

  eoValueParam<double>& tournamentRateParam = parser.createParam<double>(0.75, "tournamentRate", "Stochastic tournament rate",'R');
    double tRate = tournamentRateParam.value();

  eoValueParam<double>& sParentsElitismRateParam = parser.createParam<double>(0.1, "sParentsElitismRateParam", "Strong elitism rate for parents",'E');
    double sParentsElitismRate = sParentsElitismRateParam.value();

  eoValueParam<double>& sParentsEugenismRateParam = parser.createParam<double>(0, "sParentsEugenismRateParam", "Strong Eugenism rate",'e');
    double sParentsEugenismRate = sParentsEugenismRateParam.value();

  eoValueParam<double>& sOffspringElitismRateParam = parser.createParam<double>(0, "sOffspringElitismRateParam", "Strong elitism rate for parents",'E');
    double sOffspringElitismRate = sOffspringElitismRateParam.value();

  eoValueParam<double>& sOffspringEugenismRateParam = parser.createParam<double>(0, "sOffspringEugenismRateParam", "Strong Eugenism rate",'e');
    double sOffspringEugenismRate = sOffspringEugenismRateParam.value();

    if (parser.userNeedsHelp())
      {
        parser.printHelp(cout);
        exit(1);
      }

    unsigned i;

    cout << "Testing the replacements\nParents SIze = " << pSize 
	 << " and offspring size = " << oSize << endl;

    rng.reseed(42);


    eoDummyPop orgParents(pSize);
    eoDummyPop orgOffspring(oSize);

    // initialize so we can recognize them later!
    for (i=0; i<pSize; i++)
      orgParents[i].fitness(2*i+1);
    for (i=0; i<oSize; i++)
      orgOffspring[i].fitness(2*i);

cout << "Initial parents (odd)\n" << orgParents << "\n And initial offsprings (even)\n" << orgOffspring << endl;

    // now the ones we're going to play with
    eoDummyPop parents(0);
    eoDummyPop offspring(0);

// the replacement procedures under test
    eoGenerationalReplacement<Dummy> genReplace;
    eoPlusReplacement<Dummy> plusReplace;
    eoCommaReplacement<Dummy> commaReplace;
    eoWeakElitistReplacement<Dummy> weakElitistReplace(commaReplace);
    // the SSGA replacements
    eoSSGAWorseReplacement<Dummy> ssgaWorseReplace;
    eoSSGADetTournamentReplacement<Dummy> ssgaDTReplace(tSize);
    eoSSGAStochTournamentReplacement<Dummy> ssgaDSReplace(tRate);

    // here we go
    // Generational
    parents = orgParents;
    offspring = orgOffspring;

    cout << "eoGenerationalReplacement\n";
    cout << "=========================\n";
    genReplace(parents, offspring);
cout << "Parents (originally odd)\n" << parents << "\n And offsprings (orogonally even\n" << offspring << endl;

    // Plus
    parents = orgParents;
    offspring = orgOffspring;

    cout << "eoPlusReplacement\n";
    cout << "=================\n";
    plusReplace(parents, offspring);
cout << "Parents (originally odd)\n" << parents << "\n And offsprings (originally even)\n" << offspring << endl;

    // Comma
    parents = orgParents;
    offspring = orgOffspring;

    if (parents.size() > offspring.size() )
	cout << "Skipping Comma Replacement, more parents than offspring\n";
    else
      {
	cout << "eoCommaReplacement\n";
	cout << "==================\n";
	commaReplace(parents, offspring);
	cout << "Parents (originally odd)\n" << parents << "\n And offsprings (originally even)\n" << offspring << endl;

	// Comma with weak elitism
	parents = orgParents;
	offspring = orgOffspring;

	cout << "The same, with WEAK elitism\n";
	cout << "===========================\n";
	weakElitistReplace(parents, offspring);
	cout << "Parents (originally odd)\n" << parents << "\n And offsprings (originally even)\n" << offspring << endl;
      }

	// preparing SSGA replace worse
	parents = orgParents;
	offspring = orgOffspring;

    if (parents.size() < offspring.size() )
	cout << "Skipping all SSGA Replacements, more offspring than parents\n";
    else
      {
	cout << "SSGA replace worse\n";
	cout << "==================\n";
	ssgaWorseReplace(parents, offspring);
	cout << "Parents (originally odd)\n" << parents << "\n And offsprings (originally even)\n" << offspring << endl;

    // SSGA deterministic tournament
	parents = orgParents;
	offspring = orgOffspring;

	cout << "SSGA deterministic tournament\n";
	cout << "=============================\n";
	ssgaDTReplace(parents, offspring);
	cout << "Parents (originally odd)\n" << parents << "\n And offsprings (originally even)\n" << offspring << endl;

    // SSGA stochastic tournament
	parents = orgParents;
	offspring = orgOffspring;

	cout << "SSGA stochastic tournament\n";
	cout << "==========================\n";
	ssgaDTReplace(parents, offspring);
	cout << "Parents (originally odd)\n" << parents << "\n And offsprings (originally even)\n" << offspring << endl;
      }

    // the general replacement
    eoDeterministicSaDReplacement<Dummy> sAdReplace(sParentsElitismRate, sParentsEugenismRate, sOffspringElitismRate, sOffspringEugenismRate);// 10% parents survive

    parents = orgParents;
    offspring = orgOffspring;

    cout << "General - strong elitism\n";
    cout << "========================\n";
    sAdReplace(parents, offspring);
    cout << "Parents (originally odd)\n" << parents << "\n And offsprings (originally even)\n" << offspring << endl;


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
