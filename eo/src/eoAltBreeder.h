//-----------------------------------------------------------------------------
// eoBreeder.h
//-----------------------------------------------------------------------------

#ifndef eoBreeder_h
#define eoBreeder_h

//-----------------------------------------------------------------------------

#include <vector>          // vector
#include <iterator>
#include <eoUniform.h>     // eoUniform
#include <eoOp.h>          // eoOp, eoMonOp, eoBinOp
#include <eoPop.h>         // eoPop
#include <eoPopOps.h>      // eoTransform
#include <eoOpSelector.h>  // eoOpSelector
#include <list>
#include "eoRng.h"

using namespace std;

/*****************************************************************************
 * eoBreeder: transforms a population using genetic operators.               *
 * For every operator there is a rated to be applyed.                        *
 *****************************************************************************/

template<class EOT, class OutIt>
class eoGeneralOp: public eoOp<EOT> 
{
public:

  eoGeneralOp()
    :eoOp<EOT>( Nary ) {};
  virtual ~eoGeneralOp () {};

  virtual void operator()( vector<const EOT*> _in, OutIt _out) const = 0;
  virtual int nInputs(void) const = 0;
  virtual int nOutputs(void) const = 0; // no support for 2 -> 2 xover

   virtual string className() const {return "eoGeneralOp";};
};

template <class EOT, class OutIt>
class eoWrappedMonOp : public eoGeneralOp<EOT, OutIt>
{
public :
	eoWrappedMonOp(const eoMonOp<EOT>& _op) : eoGeneralOp<EOT, OutIt>(), op(_op) {}
	virtual ~eoWrappedMonOp() {}

  void operator()( vector<const EOT*> _in, OutIt _out) const
  {
		*_out = *_in[0];
		op(*_out );
  }
  
  int nInputs(void) const  { return 1; }
  int nOutputs(void) const { return 1; }

   virtual string className() const {return "eoWrappedOp";};


private :
	const eoMonOp<EOT>& op;
};

template <class EOT, class OutIt>
class eoWrappedBinOp : public eoGeneralOp<EOT, OutIt>
{
public :
	eoWrappedBinOp(const eoBinOp<EOT>& _op) : eoGeneralOp<EOT, OutIt>(), op(_op) {}
	virtual ~eoWrappedBinOp() {}

  void operator()( vector<const EOT*> _in, OutIt _out) const
  {
		*_out = *_in[0];
		*(_out + 1) = *_in[1];
		op(*_out, *(_out + 1));
  }
  
  int nInputs(void) const  { return 2; }
  int nOutputs(void) const { return 2; } // Yup, due to the bad design, can't choose between outputting 1 or 2 

  virtual string className() const {return "eoWrappedOp";};


private :
	const eoBinOp<EOT>& op;
};

template <class EOT, class OutIt>
class eoCombinedOp : public eoGeneralOp<EOT, OutIt>
{
public :

	eoCombinedOp() : eoGeneralOp<EOT, OutIt>(), arity(0) {}
	virtual ~eoCombinedOp() {}

	int nInputs(void) const  { return arity; }
	int nOutputs(void) const { return 1; } 

	void addOp(eoGeneralOp<EOT, OutIt>* _op)
	{
		ops.push_back(_op);
		arity = arity < _op->nInputs()? _op->nInputs() : arity;
	}


	void clear(void)
	{
		ops.resize(0);
	}


	void operator()( vector<const EOT*> _in, OutIt _out) const
	{
		for (int i = 0; i < ops.size(); ++i)
		{
			(*ops[i])(_in, _out);
			_in[0] = &*_out;
		}
	}
		
private :
	vector<eoGeneralOp<EOT, OutIt>* > ops;
	int arity;
};

template<class EOT, class OutIt>
class eoAltOpSelector: public eoOpSelector<EOT>, public vector<eoGeneralOp<EOT, OutIt>*>
{
public:
  
  virtual ID addOp( eoOp<EOT>& _op, float _arg )
  {
		eoGeneralOp<EOT, OutIt>* op = dynamic_cast<eoGeneralOp<EOT, OutIt>*>(&_op);

		
		if (op == 0)
		{
			switch(_op.readArity())
			{
			case unary :
				oplist.push_back(auto_ptr<eoGeneralOp<EOT, OutIt> >(new eoWrappedMonOp<EOT, OutIt>(static_cast<eoMonOp<EOT>&>(_op))));

				op = oplist.back().get();
				break;
			case binary :
				oplist.push_back(auto_ptr<eoGeneralOp<EOT, OutIt> >(new eoWrappedBinOp<EOT, OutIt>(static_cast<eoBinOp<EOT>&>(_op))));
				op = oplist.back().get();
				break;
			}
		}

	  iterator result = find(begin(), end(), (eoGeneralOp<EOT, OutIt>*) 0); // search for nullpointer
	  
	  if (result == end())
	  {
		  push_back(op);
		  rates.push_back(_arg);
		  return size();
	  }
	  // else
	  
		*result = op;
		ID id = result - begin();
		rates[id] = _arg;
		return id;
  }
  
  virtual const eoOp<EOT>& getOp( ID _id )
  {
	  return *operator[](_id);
  }
  
  virtual void deleteOp( ID _id )
  {
	  operator[](_id) = 0; // TODO, check oplist and clear it there too.
	  rates[_id] = 0.0;
  }
  
  virtual eoOp<EOT>* Op()
  {
	  return &selectOp();
  }


  virtual eoGeneralOp<EOT, OutIt>& selectOp() = 0;


  virtual string className() const { return "eoAltOpSelector"; };

  void printOn(ostream& _os) const {}


protected :

	vector<float> rates;
	list<auto_ptr<eoGeneralOp<EOT, OutIt> > > oplist;
};

template <class EOT, class OutIt> 
class eoProportionalOpSelector : public eoAltOpSelector<EOT, OutIt>
{
	public :
		eoProportionalOpSelector() : eoAltOpSelector<EOT, OutIt>() {}


		virtual eoGeneralOp<EOT, OutIt>& selectOp()
		{
			int what = rng.roulette_wheel(rates);

			return *operator[](what);
		}
};

template <class EOT, class OutIt> 
class eoSequentialOpSelector : public eoAltOpSelector<EOT, OutIt>
{
	public :
		
		eoSequentialOpSelector() : eoAltOpSelector<EOT, OutIt>() {}

		virtual eoGeneralOp<EOT, OutIt>& selectOp()
		{
			for (int i = 0; i < size(); ++i)
			{
				if (operator[](i) == 0)
					continue;

				if (rng.flip(rates[i]))
					combined.addOp(operator[](i));
			}

			return combined;
		}		

	private :

	eoCombinedOp<EOT, OutIt> combined;
};

template <class EOT>
class eoRandomIndy // living in a void right now
{
	public :

		eoRandomIndy() {}

	vector<const EOT*> operator()(int _n, eoPop<EOT>::iterator _begin, eoPop<EOT>::iterator _end)
	{
		vector<const EOT*> result(_n);

		for (int i = 0; i < result.size(); ++i)
		{
			result[i] = &*(_begin + rng.random(_end - _begin));
		}

		return result;
	}
};

template<class Chrom> class eoAltBreeder: public eoTransform<Chrom>
{
 public:
	 typedef eoPop<Chrom>::reverse_iterator outIt;
  /// Default constructor.
  eoAltBreeder( eoAltOpSelector<Chrom, outIt>& _opSel): opSel( _opSel ) {}
  
  /// Destructor.
  virtual ~eoAltBreeder() {}

  /**
   * Enlarges the population.
   * @param pop The population to be transformed.
   */
  void operator()(eoPop<Chrom>& pop) 
  {  
	  int size = pop.size();

      for (unsigned i = 0; i < size; i++) 
	  {
		eoGeneralOp<Chrom, outIt>& op = opSel.selectOp();

		pop.resize(pop.size() + op.nOutputs());
		vector<const Chrom*> indies = indySelector(op.nInputs(), pop.begin(), pop.begin() + size);
		
		op(indies, pop.rbegin());
	  
	  }
	}
  
  /// The class name.
  string classname() const { return "eoAltBreeder"; }
  
 private:
  eoAltOpSelector<Chrom, outIt >& opSel;
  eoRandomIndy<Chrom>    indySelector;
};



#endif eoBreeder_h
