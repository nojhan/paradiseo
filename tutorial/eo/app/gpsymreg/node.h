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

#ifndef _NODE_H
#define _NODE_H



#include <iostream>
#include <string>
#include <cmath> // for finite(double) function

using namespace gp_parse_tree;
using namespace std;


/* A new Operation and Node class for even more flexibility.

   Improvements over the t-eoSymreg code are:

   * No hardcoded functions or operators. The Operation and Node class below
     allow you to specify your own unary and binary functions as well as
     binary operators (like +,-,*,/). Moreover you can detemine if you want
     to allow primitve subroutines with either one or two arguments.

     If a Node has a subroutine Operation it will take evaluate the first
     (and possible second) child branch and use them as input variables for
     the remaining second (or third) child branch.
*/


typedef enum {Variable, UFunction, BFunction, BOperator, Const} Type;

typedef double (*BinaryFunction)(const double,const double);
typedef double (*UnaryFunction)(const double);

struct Operation
{
	public:

		typedef unsigned int VariableID;
		typedef string Label;


		// if your compiler allows you to have nameless unions you can make this a
		// union by removing the //'s below

		//union
		//{
			UnaryFunction uFunction;
			BinaryFunction bFunction;
			VariableID id;
			double constant;
		//};



		Label label;
		Type type;

		// the default constructor results in a constant with value 0
		Operation() : constant(0),  label("0"), type(Const){};
		// two possible constructors for Unary Functions
		Operation(UnaryFunction _uf, Label _label): uFunction(_uf), label(_label), type(UFunction) {};
		Operation(Label _label, UnaryFunction _uf): uFunction(_uf), label(_label), type(UFunction) {};

		// Watch out there are two constructors using pointers two binary functions:
		// Binary Function (printed as label(subtree0,subtree1)  (e.g. pow(x,y))
		// Binary Operator (printed as (subtree0 label subtree1) (e.g. x^y)
		// The difference is purely cosmetic.

		// If you specify the label before the function pointer -> Binary Function
		Operation(Label _label, BinaryFunction _bf): bFunction(_bf), label(_label), type(BFunction) {};
		// If you specify the function pointer before the label -> Binary Operator
		Operation(BinaryFunction _bf, Label _label): bFunction(_bf), label(_label), type(BOperator) {};

		// A constructor for variables
		Operation(VariableID _id, Label _label): id(_id), label(_label), type(Variable) {};
		// A constructor for constants
		Operation(double _constant, Label _label): constant(_constant), label(_label), type(Const) {};


		Operation(const Operation &_op)
		{
			switch(_op.type)
			{
				case Variable: id = _op.id; break;
				case UFunction: uFunction = _op.uFunction;  break;
				case BFunction: bFunction = _op.bFunction; break;
				case BOperator: bFunction = _op.bFunction; break;
				case Const: constant = _op.constant;  break;
			}
			type = _op.type;
			label = _op.label;
		};
		virtual ~Operation(){};

};


class Node
{
	private:
		Operation op;

	public:

		Node(void): op(Operation()){};
		Node(Operation &_op) : op(_op){};
		virtual ~Node(void) {}

		int arity(void) const
		{
			switch(op.type)
			{
				case Variable: return 0;
				case UFunction: return 1;
				case BFunction: return 2;
				case BOperator: return 2;
				case Const: return 0;
			}
			return 0;
		}

		void randomize(void) {}

		template<class Children>
		void operator()(double& result, Children args, vector<double> &var) const
		{
			double result0;
			double result1;


			switch(op.type)
			{
				case Variable:  result = var[op.id%var.size()]; //%var.size() used in the case of Subroutines and as a security measure
						break;
				case UFunction: args[0].apply(result0, var);
						result = op.uFunction(result0);
						break;
				case BFunction:
				case BOperator:	args[0].apply(result0, var);
						args[1].apply(result1, var);
						result = op.bFunction(result0,result1);
						break;
				case Const:	result = op.constant;
						break;

			}

		}

		template<class Children>
		void operator()(string& result, Children args) const
		{

			string subtree0;
			string subtree1;
			string subtree2;

			switch(op.type)
			{

				case Variable:
				case Const: 	result += op.label;
						break;

				case UFunction:	result += op.label;
						result += "(";
						args[0].apply(subtree0);
						result += subtree0;
						result += ")";
						break;
				case BFunction: result += op.label;
						result += "(";
						args[0].apply(subtree0);
						result += subtree0;
						result += ",";
						args[1].apply(subtree1);
						result += subtree1;
						result += ")";
						break;
				case BOperator: result += "(";
						args[0].apply(subtree0);
						result += subtree0;
						result += op.label;
						args[1].apply(subtree1);
						result += subtree1;
						result += ")";
						break;
				default: result += "ERROR in Node::operator(string,...) \n"; break;
			}
		}

		Operation getOp(void) const {return op;}

};











//-----------------------------------------------------------
// saving, loading LETS LEAVE IT OUT FOR NOW



std::ostream& operator<<(std::ostream& os, const Node& eot)
{
    Operation op(eot.getOp());

    os << (eot.getOp()).label;
    return os;
}


// we can't load because we are using function pointers. Instead we prevent a compiler warning by calling the arity() function.
std::istream& operator>>(std::istream& is, Node& eot)
{
    eot.arity();
    return is;
}



#endif
