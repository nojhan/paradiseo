/*
    This library is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.
    
    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.
 
    You should have received a copy of the GNU General Public License
    along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 
    Contact: todos@geneura.ugr.es, http://geneura.ugr.es
             jeggermo@liacs.nl
*/

#ifndef _FITNESS_FUNCTION_H
#define _FITNESS_FUNCTION_H

#include <gp/eoParseTree.h>
#include <eo>

#include <cmath>	
#include "parameters.h"
#include "node.h"

using namespace gp_parse_tree;
using namespace std;



// the first fitness is the normal goal fitness
// the second fitness is the tree size (we prefer smaller trees)
// lets use names to define the different fitnesses
#define NORMAL 0      // Stepwise Adaptation of Weights Fitness
#define SMALLESTSIZE 1  // The size of the tree, we want to minimize this one -- statistics will tell us the smallest tree size


// Look: overloading the maximization without overhead (thing can be inlined)
class MinimizingFitnessTraits : public eoParetoFitnessTraits
{
  public :
  static bool maximizing(int which) { return false;} // we want to minimize both fitnesses}
  static unsigned nObjectives()          { return 2;} // the number of fitnesses }
};

// Lets define our MultiObjective FitnessType
typedef eoParetoFitness<MinimizingFitnessTraits> FitnessType;


// John Koza's sextic polynomial (our example problem)

double sextic_polynomial(double x)
{
	double result=0;
	result = pow(x,6) - (2*pow(x,4)) + pow(x,2);
	return result;
};

// we use the following functions for the basic math functions

double _plus(double arg1, double arg2)
{
	return arg1 + arg2;
}

double _minus(double arg1, double arg2)
{
	return arg1 - arg2;
}

double _multiplies(double arg1, double arg2)
{
	return arg1 * arg2;
}

double _divides(double arg1, double arg2)
{
	return arg1 / arg2;
}

double _negate(double arg1)
{
	return -arg1;
}	



// now let's define our tree nodes

template<class TreeNode>
void init(vector<TreeNode> &initSequence)
{
			
		// we have only one variable (X)
		Operation varX( (unsigned int) 0, string("X") );
			
			
		// the main binary operators  
		Operation OpPLUS ( _plus, string("+"));
		Operation OpMINUS( _minus,string("-"));
		Operation OpMULTIPLIES(_multiplies,string("*"));
		// We can use the normal divide function because there is a check for finite numbers in the node class
		// so PDIV (protected divided) is enforced there so: (x/0 -> nan -> 0)
		Operation OpDIVIDE( _divides, string("/") );
		// we can also use the standard 'pow' function from cmath or math because of the check for nan is
		// in the node class so: (-3^3.1) -> nan -> 0)
		Operation OpPOW( pow, string("^") );
		
		
		// Now the functions as binary functions
		Operation PLUS( string("plus"), _plus);
		Operation MINUS( string("minus"), _minus);
		Operation MULTIPLIES( string("multiply"), _multiplies);
		Operation DIVIDE( string("divide"), _divides);
		Operation POW(string("pow"), pow);
		
		
		// and some unary  functions
		Operation NEGATE( _negate,string("-"));
		Operation SIN ( sin, string("sin"));
		Operation COS ( cos, string("cos"));
		// all functions are "protected" inside the Node class so can also use tan(x)
		// resulting values of -inf, inf or NaN (not-a-number) are converted to 0
		Operation TAN ( tan, string("tan"));
		Operation EXP ( exp, string("e^"));
		Operation LOG ( log, string("ln"));
		
			
		// Now we are ready to add the possible nodes to our initSequence (which is used by the eoDepthInitializer)
			
		// always add the leaves (nodes with arity 0) first (or the program will crash)
		// so lets start with our variable
		initSequence.push_back(varX);
			
		// followed by the constants 2, 4, 6
		for(unsigned int i=2; i <= 6; i+=2)
		{
			char text[255];
			sprintf(text, "%i", i);
			Operation op(i*1.0, text);
			initSequence.push_back( op );
			// and we add the variable again (so we have get lots of variables);
			initSequence.push_back( varX );
		}	
			
		// next we add the unary functions
		/*	
		initSequence.push_back( NEGATE );
		initSequence.push_back( SIN );
		initSequence.push_back( COS );
		initSequence.push_back( TAN );
		initSequence.push_back( EXP );
		initSequence.push_back( LOG );
		
		// and the binary functions
		initSequence.push_back( PLUS);
		initSequence.push_back( MINUS );
		initSequence.push_back( MULTIPLIES );
		initSequence.push_back( DIVIDE );
		initSequence.push_back( POW );
		*/
		// and the binary operators
		initSequence.push_back( OpPLUS);
		initSequence.push_back( OpMINUS );
		/*
		initSequence.push_back( OpMULTIPLIES );
		initSequence.push_back( OpDIVIDE );
		*/
		initSequence.push_back( OpPOW );
		
			
};


template <class FType, class TreeNode> 
class RegFitness: public eoEvalFunc< eoParseTree<FType, TreeNode> > 
{
	public:
	
    		typedef eoParseTree<FType, TreeNode> EoType; 
    		
		void operator()(EoType &_eo)
		{
		
				vector< double > input(1); // the input variable(s)
				double output;
				double target;
				FType fitness;
				
				
				float x=0;
				double fit=0;	
				for(x=-1; x <= 1; x+=0.1)
				{
					input[0] = x;
					target = sextic_polynomial(x);
					_eo.apply(output,input);
					
					fit += pow(target - output, 2);
				}
				// check if the fitness is valid
				// some versions of gcc (e.g. 2.95.2 on solaris) don't have isinf(x) defined
				#ifdef isinf
				if (isinf(fit) == 0)
					fitness[NORMAL] = fit;
				else
					fitness[NORMAL] = MAXFLOAT;	
				#endif
				
				fitness[SMALLESTSIZE] =  _eo.size() / (1.0*parameter.MaxSize);
				_eo.fitness(fitness);
				
				if (fitness[NORMAL] < best[NORMAL])
				{
					best[NORMAL] = fitness[NORMAL];
					tree="";
					_eo.apply(tree);
				}	
						
		}
		
		
		
		RegFitness(eoValueParam<unsigned> &_generationCounter, vector< TreeNode > &initSequence, Parameters &_parameter) : eoEvalFunc<EoType>(), generationCounter(_generationCounter), parameter(_parameter) 
		{
			init<TreeNode>(initSequence);
			best[NORMAL] = 1000;
			tree= "not found";
		};
		
	   	~RegFitness()
		{
			cerr << "Best Fitness= " << best[NORMAL] << endl;
			cerr << tree << endl;
		};

	private:
    		eoValueParam<unsigned> &generationCounter; // so we know the current generation
		Parameters &parameter; // the parameters
		FType best;	// the best found fitness
		string tree;	
};

#endif

