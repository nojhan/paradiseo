#pragma warning(disable:4786)

#include "eoParseTree.h"
#include "eoEvalFunc.h"

using namespace gp_parse_tree;
using namespace std;

//-----------------------------------------------------------------------------

class SymregNode
{
public :

	enum Operator {X = 'x', Plus = '+', Min = '-', Mult = '*', PDiv = '/'};

	SymregNode(void)				{ init(); }
	SymregNode(Operator _op)		{ op = _op; }
	virtual ~SymregNode(void)			{}

	// arity function
	int arity(void) const      { return op == X? 0 : 2; }
	
	// evaluation function, single case, using first argument to give value of variable
	template <class Children>
	double operator()(double var, Children args) const
	{
		switch (op)
		{
		case Plus : return args[0].apply(var) + args[1].apply(var);
		case Min  : return args[0].apply(var) - args[1].apply(var);
		case Mult : return args[0].apply(var) * args[1].apply(var);
		case PDiv : 
			{
				double arg1 = args[1].apply(var);
				if (arg1 == 0.0)
					return 1.0; // protection a la Koza, realistic implementations should maybe throw an exception

				return args[0].apply(var) / arg1;
			}

		case X    : return var; 
		}

        return var; // to avoid compiler error
	}

    /// 'Pretty' print to ostream function
    template <class Children>
        string operator()(string dummy, Children args)
    {
        static const string lb = "(";
        static const string rb = ")";
        char opStr[4] = "   ";
        opStr[1] = op;
        
	    if (arity() == 0)
		{
            return string("x");
        }
        // else
        string result = lb + args[0].apply(dummy);
        result += opStr; 
        result += args[1].apply(dummy) + rb;
        return result;
    }

    Operator getOp(void) const { return op; }    

protected :

	void init(void)					{ op = X; }

private :

	Operator op; // the type of node
};

/// initializor
static SymregNode init_sequence[5] = {SymregNode::X, SymregNode::Plus, SymregNode::Min, SymregNode::Mult, SymregNode::PDiv}; // needed for intialization

//-----------------------------------------------------------
// saving, loading

std::ostream& operator<<(std::ostream& os, const SymregNode& eot)
{
    os << static_cast<char>(eot.getOp());
    return os;
}

std::istream& operator>>(std::istream& is, SymregNode& eot)
{
    char type;
    type = (char) is.get();
    eot = SymregNode(static_cast<SymregNode::Operator>(type));
    return is;
}


//-----------------------------------------------------------------------------
/** Implementation of a function evaluation object. */

float targetFunction(float x)
{ 
	return x * x * x * x - x * x * x + x * x * x - x * x + x - 1; 
}

// parameters controlling the sampling of points
const float xbegin = -10.0f;
const float xend   = 10.0f;
const float xstep  = 1.3f; 

template <class FType, class Node> struct RMS: public eoEvalFunc< eoParseTree<FType, Node> > 
{
public :

    typedef eoParseTree<FType, Node> EoType;

    typedef eoParseTree<FType, Node> argument_type;
    typedef double                   fitness_type;

	RMS(void) : eoEvalFunc<EoType>()
	{
		int n = int( (xend - xbegin) / xstep);
		
		inputs.resize(n);
		target.resize(n);

		int i = 0;

    	for (double x = xbegin; x < xend && i < n; ++i)
		{
			target[i] = targetFunction(x);
			inputs[i] = x;
		}
	}

    ~RMS() {}
    
	void operator()( EoType & _eo ) const 
	{
		vector<double> outputs;
		outputs.resize(inputs.size());
		
        double fitness = 0.0;
        
		for (int i = 0; i < inputs.size(); ++i)
        {
		    outputs[i] = _eo.apply(inputs[i]);
	        fitness += (outputs[i] - target[i]) * (outputs[i] - target[i]);
        }
		
        fitness /= (double) target.size();
        fitness = sqrt(fitness);
			
		if (fitness > 1e+20)
			fitness = 1e+20;

		_eo.fitness(fitness);
	}

private :
	vector<double> inputs;
	vector<double> target;
};

#include "eoTerm.h"

template <class EOT>
class eoGenerationTerm : public eoTerm<EOT>
{
    public :
        eoGenerationTerm(size_t _ngen) : eoTerm<EOT>(), ngen(_ngen) {}

        bool operator()(const eoPop<EOT>&)
        {
            cout << '.'; // pacifier
            cout.flush();

            return --ngen > 0;
        }
    private :
        unsigned ngen;
};

template <class EOT, class FitnessType>
void print_best(eoPop<EOT>& pop)
{
    cout << endl;
    FitnessType best = pop[0].fitness();
    int index = 0;

    for (int i = 1; i < pop.size(); ++i)
    {
        if (best < pop[i].fitness())
        {
            best = pop[i].fitness();
            index = i;
        }
    }
    
    cout << "\t";
        
    string str = pop[index].apply(string());
    
    cout << str.c_str();
    cout << endl << "Error = " << pop[index].fitness() << endl;
}


#include <eo>
#include "eoGOpBreeder.h"
#include "eoSequentialGOpSelector.h"
#include "eoProportionalGOpSelector.h"
#include "eoDetTournamentIndiSelector.h"
#include "eoDetTournamentInserter.h"
#include "eoSteadyStateEA.h"
#include "eoScalarFitness.h"

void main()
{
    typedef eoScalarFitness<double, greater<double> > FitnessType;
    typedef SymregNode GpNode;

    typedef eoParseTree<FitnessType, GpNode> EoType;
    typedef eoPop<EoType> Pop;	

    const int MaxSize = 75;
    const int nGenerations = 50;

    // Initializor sequence, contains the allowable nodes
    vector<GpNode> init(init_sequence, init_sequence + 5);

    // Depth Initializor, defaults to grow method.
    eoGpDepthInitializer<FitnessType, GpNode> initializer(10, init);
    
    // Root Mean Squared Error Measure
    RMS<FitnessType, GpNode>              eval;

    Pop pop(500, MaxSize, initializer, eval);

    eoSubtreeXOver<FitnessType, GpNode>   xover(MaxSize);
    eoBranchMutation<FitnessType, GpNode> mutation(initializer, MaxSize);

    eoSequentialGOpSel<EoType>   seqSel;

    seqSel.addOp(mutation, 0.25);
    seqSel.addOp(xover, 0.75);
  
    eoDetTournamentIndiSelector<EoType> selector(5);
  
    eoDetTournamentInserter<EoType> inserter(eval, 5);
  
    // Terminators
    eoGenerationTerm<EoType> term(nGenerations);

    // GP generation
    eoSteadyStateEA<EoType> gp(seqSel, selector, inserter, term);

    cout << "Initialization done" << endl;

    print_best<EoType, FitnessType>(pop);

    try
    {
      gp(pop);
    }
    catch (exception& e)
    {
	    cout << "exception: " << e.what() << endl;;
	    exit(EXIT_FAILURE);
    }

    print_best<EoType, FitnessType>(pop);
}

