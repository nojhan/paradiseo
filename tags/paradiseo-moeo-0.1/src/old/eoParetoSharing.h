// -*- C++ -*-

#include <EO.h>
#include <eoPerf2Worth.h>
#include <old/eoParetoPhenDist.h>
#include <eoParetoRanking.h>

template < class EOT, class worthT =
  double >class eoParetoSharing:public eoPerf2Worth < EOT, worthT >
{
public:

  eoParetoSharing (double _nicheSize):eoPerf2Worth < EOT,
    worthT > ("ParetoSharing"), nicheSize (_nicheSize), dist (euc_dist),
    Dmax (_nicheSize)
  {
  }


  eoParetoSharing (double _nicheSize, eoParetoPhenDist < EOT,
		   worthT > &_dist):eoPerf2Worth < EOT,
    worthT > ("ParetoSharing"), nicheSize (_nicheSize), dist (_dist),
    Dmax (_nicheSize)
  {
  }




  void operator () /*calculate_worths */ (const eoPop < EOT > &_pop)
  {

    unsigned i, j, pSize = _pop.size ();
    //cout<<"**************************************************************************************\n";
    // std :: cout << "psize = " << pSize << std :: endl ;
    if (pSize <= 1)
      throw std::
	runtime_error ("Apptempt to do sharing with population of size 1");
    eoPerf2Worth < EOT, worthT >::value ().resize (pSize);
    std::vector < double >sim (pSize);	// to hold the similarities

    dMatrix distMatrix (pSize);

// compute the distance
    distMatrix[0][0] = 0;
    for (i = 1; i < pSize; i++)
      {
	distMatrix[i][i] = 0;
	for (j = 0; j < i; j++)
	  {
	    //if
	    distMatrix[i][j] = distMatrix[j][i] = dist (_pop[i], _pop[j]);
	    //cout<<"   "<<distMatrix[j][i]<<"  "<<dist(_pop[i],_pop[j])<<"\n";
	  }

      }

//compute the similarities
    for (i = 0; i < pSize; i++)
      {
	double sum = 0.0;
	for (j = 0; j < pSize; j++)

	  sum += sh (distMatrix[i][j], Dmax);
	sim[i] = sum;

//cout<<"\n  i  ----->"<<sim[i]<<"\n";
      }

    eoDominanceMap < EOT > Dmap1;
    Dmap1.setup (_pop);

    eoParetoRanking < EOT > rnk1 (Dmap1);
    rnk1.calculate_worths (_pop);
// now set the worthes values
    for (i = 0; i < pSize; ++i)
      {
	typename EOT::Fitness v;



//cout<<"voila: "<<
//rnk1.value().operator[](i);

//vector<double> v;
//v.resize(_pop[i].fitness().size());
//for(unsigned k=0;k<_pop[i].fitness().size();++k)
//v[k]=_pop[i].fitness().operator[](k)/sim[i];
//_pop[i].fitness(v);//.operator[](k)=0;//_pop[i].fitness().operator[](k)/sim[i];
	eoPerf2Worth < EOT, worthT >::value ()[i] = rnk1.value ().operator[](i) / sim[i];	//*_pop[i].fitness().operator[](1)*_pop[i].fitness().operator[](1));
//cout<<"\n__________"<<pSize<<"  "<<value()[i]<<"    "<<i;
      }

  }




  class dMatrix:public std::vector < std::vector < double > >
  {
  public:
    dMatrix (unsigned _s):rSize (_s)
    {
      this->resize (_s);
      for (unsigned i = 0; i < _s; ++i)
	this->operator[] (i).resize (_s);
    }

    void printOn (std::ostream & _os)
    {
      for (unsigned i = 0; i < rSize; i++)
	for (unsigned j = 0; j < rSize; ++j)
	  {
	    _os << this->operator[](i)[j] << " ";
	    _os << endl;
	  }
      _os << endl;
    }
//public:
//std::vector<double>v;

  private:




    unsigned rSize;
  };

private:

  ;

  double sh (double dist, double Dmax)
  {
    if (dist < Dmax)
      return (1.0 - dist / Dmax);
    else
      return (0.0);
  }

  double nicheSize;
  eoParetoPhenDist < EOT, worthT > &dist;
  eoParetoEuclidDist < EOT > euc_dist;
  double Dmax;

};
