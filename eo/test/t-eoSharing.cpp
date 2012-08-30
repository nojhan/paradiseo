#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

// to avoid long name warnings
#ifdef _MSC_VER
#pragma warning(disable:4786)
#endif

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

// general
#include <eo>
#include <utils/eoDistance.h>

//-----------------------------------------------------------------------------

struct Dummy : public EO<double>
{
    typedef double Type;
  void printOn(std::ostream & _os) const
  {
      EO<double>::printOn(_os);
      std::cout << " " << xdist ;
  }
  double xdist;
};

class
eoDummyDistance : public eoDistance<Dummy>
{
  double operator()(const Dummy & _v1, const Dummy & _v2)
  {
    double r= _v1.xdist - _v2.xdist;
    return sqrt(r*r);
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

    //    cout << "Pop offspring \n" << offspring << endl;

    // compute stats
    std::vector<unsigned> nb(parents.size(), 0);
    for (i=0; i<offspring.size();  i++)
      {
	unsigned trouve = isInPop<Dummy>(offspring[i], parents);
	if (trouve == parents.size()) // pas trouve
	  throw std::runtime_error("Pas trouve ds parents");
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

  // random seed
    eoValueParam<uint32_t>& seedParam = parser.createParam(uint32_t(0), "seed", "Random number seed", 'S');
    if (seedParam.value() == 0)
	seedParam.value() = time(0);
    rng.reseed(seedParam.value());


  // pSize global variable !
  eoValueParam<unsigned> pSizeParam = parser.createParam(unsigned(10), "parentSize", "Parent size",'P');
  pSize = pSizeParam.value();

  eoHowMany oRate = parser.createParam(eoHowMany(1.0), "offsrpringRate", "Offsrpring rate (% or absolute)",'O').value();

  eoHowMany fRate = parser.createParam(eoHowMany(1.0), "fertileRate", "Fertility rate (% or absolute)",'F').value();


  double nicheSize = parser.createParam(0.1, "nicheSize", "Paramter Sigma for Sharing",'\0').value();

  eoParamParamType & peakParam = parser.createParam(eoParamParamType("2(1,2)"), "peaks", "Description of the peaks: N(nb1,nb2,...,nbN)", 'p').value();

  // the number of peaks: first item of the paramparam
  unsigned peakNumber = atoi(peakParam.first.c_str());
  if (peakNumber < 2)
      {
	std::cerr << "WARNING, nb of peaks must be larger than 2, using 2" << std::endl;
	peakNumber = 2;
      }

  std::vector<unsigned> nbIndiPerPeak(peakNumber);
  unsigned i, sum=0;

  // the second item is a vector<string> containing all values
  if (!peakParam.second.size())   // no other parameter : equal peaks
      {
	std::cerr << "WARNING, no nb of indis per peaks, using equal nbs" << std::endl;
	for (i=0; i<peakNumber; i++)
	  nbIndiPerPeak[i] = pSize/peakNumber;
      }
    else	  // parameters passed by user
      if (peakParam.second.size() != peakNumber)
	{
	  std::cerr << "ERROR, not enough nb of indis per peaks" << std::endl;
	  exit(1);
	}
      else    // now we have in peakParam.second all numbers
	{
	  for (i=0; i<peakNumber; i++)
	    sum += ( nbIndiPerPeak[i] = atoi(peakParam.second[i].c_str()) );
	  // now normalize
	  for (i=0; i<peakNumber; i++)
	    nbIndiPerPeak[i] = nbIndiPerPeak[i] * pSize / sum;
	}

  // compute exact total
  sum = 0;
  for (i=0; i<peakNumber; i++)
    sum += nbIndiPerPeak[i];
  if (sum != pSize)
    {
      pSize = pSizeParam.value() = sum;
      std::cerr << "WARNING, adjusting pSize to " << pSize << std::endl;
    }

    make_help(parser);

    // hard-coded directory name ...
    std::cout << "Testing the Sharing\n";
    std::cout << " There will be " << peakNumber << " peaks";
    std::cout << " with respective pops ";
    for (i=0; i<peakNumber; i++)
      std::cout << nbIndiPerPeak[i] << ", ";
    std::cout << "\n Peaks are at distance 1 from one-another (dim 1),\n";
      std::cout << " fitness of each peak is nb of peak, and\n";
      std::cout << " fitness of individuals = uniform[fitness of peak +- 0.01]\n\n";

      std::cout << "The resulting file (in dir ResSelect), contains \n";
    std::cout << " the empirical proba. for each indi to be selected." << std::endl;
    system("mkdir ResSelect");

    // initialize parent population
    parentsOrg.resize(pSize);

    // all peaks of equal size in fitness, with different nn of individuals
    unsigned index=0;
    for (unsigned nbP=0; nbP<peakNumber; nbP++)
      for (i=0; i<nbIndiPerPeak[nbP]; i++)
	{
	  parentsOrg[index].fitness(nbP+1 + 0.02*eo::rng.uniform() - 0.01);
	  parentsOrg[index].xdist = nbP+1 + 0.02*eo::rng.uniform() - 0.01;
	  index++;
	}

    std::cout << "Initial population\n" << parentsOrg << std::endl;

    char fileName[1024];

// the selection procedures under test
    //    eoDetSelect<Dummy> detSelect(oRate);
    //    testSelectMany(detSelect, "detSelect");

    // Sharing using the perf2Worth construct
    // need a distance for that
    eoDummyDistance dist;
    eoSharingSelect<Dummy> newSharingSelect(nicheSize, dist);
    sprintf(fileName,"Niche_%g",nicheSize);
    testSelectOne<Dummy>(newSharingSelect, oRate, fRate, fileName);

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
