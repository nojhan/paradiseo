//-----------------------------------------------------------------------------
// eoBreeder.h
//-----------------------------------------------------------------------------

#ifndef eoBreeder_h
#define eoBreeder_h

//-----------------------------------------------------------------------------

#include <vector>  // vector

using namespace std;

#include <eo>      // eoTransform, eoUniform, eoOp, eoMonOp, eoBinOp, eoPop

/*****************************************************************************
 * eoBreeder: transforms a population using genetic operators.               *
 * For every operator there is a rated to be applyed.                        *
 *****************************************************************************/

template<class Chrom> class eoBreeder: public eoTransform<Chrom>
{
 public:
  /// Default constructor.
  eoBreeder( eoOpSelector<Chrom>& _opSel): opSel( _opSel ) {}
  

  /// Destructor.
  virtual ~eoBreeder() { }

  /**
   * Transforms a population.
   * @param pop The population to be transformed.
   */

  void operator()(eoPop<Chrom>& pop) 
    {
      for (unsigned i = 0; i < pop.size(); i ++ ) {
		  eoOp<Chrom>* op = opSel.Op();
		switch (op->readArity()) {
			case unary:
			{
				eoMonOp<Chrom>* monop = static_cast<eoMonOp<Chrom>* >(op);
				(*monop)( pop[i] );
				break;
			}
			case binary:
			{
				eoBinOp<Chrom>* binop = 
					static_cast<eoBinOp<Chrom>* >(op);
				eoUniform<unsigned> u(0, pop.size() );
				(*binop)(pop[i], pop[ u() ] );
				break;
			}
			case Nary:
			{
				eoNaryOp<Chrom>* Nop = 
					static_cast<eoNaryOp<Chrom>* >(op);
				eoUniform<unsigned> u(0, pop.size() );
				eoPop<Chrom> tmpVec;
				tmpVec.push_back( pop[i] );
				for ( unsigned i = 0; i < u(); i ++ ) {
					tmpVec.push_back( pop[ u() ] );
				}
				(*Nop)( tmpVec );
				break;
			}
			default:
			{
				throw runtime_error( "eoBreeder:operator() Not implemented yet!" );
				break;
			}
	    }
    }
  };
  
  /// The class name.
  string classname() const { return "eoBreeder"; }
  
 private:
  eoOpSelector<Chrom>& opSel;
  
};

//-----------------------------------------------------------------------------

#endif eoBreeder_h
