//-----------------------------------------------------------------------------
// eoBreeder.h
//-----------------------------------------------------------------------------

#ifndef eoBreeder_h
#define eoBreeder_h

//-----------------------------------------------------------------------------

#include <vector>  // vector
#include <pair.h>  // pair
#include <eo>      // eoTransform, eoUniform, eoOp, eoMonOp, eoBinOp, eoPop

/*****************************************************************************
 * eoBreeder: transforms a population using genetic operators.               *
 * For every operator there is a rated to be applyed.                        *
 *****************************************************************************/

template<class Chrom> class eoBreeder: public eoTransform<Chrom>
{
 public:
  /// Default constructor.
  eoBreeder(): uniform(0.0, 1.0) {}
  
  /**
   * Adds a genetic operator.
   * @param operator The operator.
   * @param rate Rate to apply the operator.
   */
  void add(eoOp<Chrom>& op, float rate = 1.0)
    {
      operators.push_back(pair<float, eoOp<Chrom>*>(rate, &op));
    }
  
  /**
   * Transforms a population.
   * @param pop The population to be transformed.
   */
  void operator()(eoPop<Chrom>& pop)
    {
      unsigned i;

      for (Operators::const_iterator op = operators.begin(); 
	   op != operators.end(); 
	   op++)
	switch (op->second->readArity())
	  {
	  case unary:
	    { 
	      eoMonOp<Chrom>& monop = (eoMonOp<Chrom>&)*(op->second);	
	      for (i = 0; i < pop.size(); i++)
		if (uniform() < op->first)
		  monop(pop[i]);  
	      break;
	    }
	  case binary:
	    {
	      vector<unsigned> pos(pop.size());
	      iota(pos.begin(), pos.end(), 0);
	      random_shuffle(pos.begin(), pos.end());
	      
	      cout << "pos = ";
	      copy(pos.begin(), pos.end(),
		   ostream_iterator<unsigned>(cout, " "));
	      cout << endl;

	      eoBinOp<Chrom>& binop = (eoBinOp<Chrom>&)*(op->second);
	      for (i = 1; i < pop.size(); i += 2)
		if (uniform() < op->first)
		  binop(pop[pos[i - 1]], pop[pos[i]]);
	      break;
	    }
	  default:
	    {
	      cerr << "eoBreeder:operator() Not implemented yet!" << endl;
	      exit(EXIT_FAILURE);
	      break;
	    }
	  }
    }
  
  /// The class name.
  string classname() const { return "eoBreeder"; }
  
 private:
  typedef vector<pair<float, eoOp<Chrom>*> > Operators;

  eoUniform<float> uniform;
  Operators  operators;
};

//-----------------------------------------------------------------------------

#endif eoBreeder_h
