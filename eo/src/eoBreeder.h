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
  
  /// Destructor.
  ~eoBreeder()
    {
      for (Operators::iterator op = operators.begin();
	   op != operators.end();
	   op++)
	delete op->second;
    }
  
  /**
   * Adds a genetic operator.
   * @param operator The operator.
   * @param rate Rate to apply the operator.
   */
  void add(eoOp<Chrom>* operator, float rate = 1.0)
    {
      operators.push_back(pair<float, eoOp<Chrom> * >(operator, rate));
    }
  
  /**
   * Transforms a population.
   * @param pop The population to be transformed.
   */
  void operator()(eoPop<Chrom>& pop) const
    {
      for (Operators::const_iterator op = operators.begin(); 
	   op != operators.end(); 
	   op++)
	if (op->first < uniform())
	  switch (op->second->readArity())
	    {
	    case unary:
	      {
		eoMonOp<Chrom>& monop = 
		  dinamic_cast<eoMonOp<Chrom> >(*(op->second));
		for_each(pop.begin(), pop.end(), monop);
		break;
	      }
	    case binary:
	      {
		eoBinOp<Chrom>& binop = 
		  dinamic_cast<eoBinOp<Chrom> >(*(op->second));
		for (unsigned i = 1; i < pop.size(); i += 2)
		  binop(pop[i - 1], pop[i]);
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
