#ifndef EO_PARSE_TREE_H
#define EO_PARSE_TREE_H

#include <list>

#include <EO.h>
#include <eoOp.h>
#include <gp/parse_tree.h>
#include <eoInit.h>

using namespace gp_parse_tree;
using namespace std;

template <class FType, class Node>
class eoParseTree : public EO<FType>, public parse_tree<Node>
{
public :

    typedef parse_tree<Node>::subtree Subtree;

    eoParseTree(void) {}
    eoParseTree(const parse_tree<Node>& tree) : parse_tree<Node>(tree) {}
    
    virtual void pruneTree(unsigned _size)
    {
        if (_size < 1)
            return;

        while (size() > _size)
        {
            back() = operator[](size()-2); 
        }
    }

    eoParseTree(std::istream& is) : EO<FType>(), parse_tree<Node>() 
    {
        readFrom(is);
    }

    string className(void) const { return "eoParseTree"; }

    void printOn(std::ostream& os) const
    {
        os << fitness() << ' ';

        std::copy(ebegin(), eend(), ostream_iterator<Node>(os));
    }

    void readFrom(std::istream& is) 
    {
        FType fit;

        is >> fit;

        fitness(fit);

        std::copy(istream_iterator<Node>(is), istream_iterator<Node>(), back_inserter(*this));
    }
};

template <class FType, class Node>
std::ostream& operator<<(std::ostream& os, const eoParseTree<FType, Node>& eot)
{
    eot.printOn(os);
    return os;
}

template <class FType, class Node>
std::istream& operator>>(std::istream& is, eoParseTree<FType, Node>& eot)
{
    eot.readFrom(is);
    return is;
}


template <class FType, class Node>
class eoGpDepthInitializer : public eoInit< eoParseTree<FType, Node> >
{
    public :

    typedef eoParseTree<FType, Node> EoType;

	eoGpDepthInitializer(
        unsigned _max_depth,
		const vector<Node>& _initializor,
        bool _grow = true) 
            :
            eoInit<EoType>(),
			max_depth(_max_depth),
			initializor(_initializor),
			grow(_grow) 
    {}

	virtual string className() const { return "eoDepthInitializer"; };

    void operator()(EoType& _tree)
	{
        list<Node> sequence;
        
        generate(sequence, max_depth);
        
        parse_tree<Node> tmp(sequence.begin(), sequence.end());
        _tree.swap(tmp);
	}

    void generate(list<Node>& sequence, int the_max, int last_terminal = -1)
    {
	    if (last_terminal == -1)
	    { // check where the last terminal in the sequence resides
            vector<Node>::iterator it;
		    for (it = initializor.begin(); it != initializor.end(); ++it)
		    {
			    if (it->arity() > 0)
				    break;
		    }
		
		    last_terminal = it - initializor.begin();
	    }

	    if (the_max == 1)
	    { // generate terminals only
		    vector<Node>::iterator it = initializor.begin() + rng.random(last_terminal);
		    sequence.push_front(*it);
		    return;
	    }
	
	    vector<Node>::iterator what_it;

	    if (grow)
	    {
		    what_it = initializor.begin() + rng.random(initializor.size());
	    }
	    else // full
	    {
		    what_it = initializor.begin() + last_terminal + rng.random(initializor.size() - last_terminal);
	    }

        what_it->randomize();

	    sequence.push_front(*what_it);

	    for (int i = 0; i < what_it->arity(); ++i)
		    generate(sequence, the_max - 1, last_terminal);
    }


private :

	unsigned max_depth; 
    std::vector<Node> initializor;
	bool grow; 
};

template<class FType, class Node>
class eoSubtreeXOver: public eoQuadOp< eoParseTree<FType, Node> > {
public:

  typedef eoParseTree<FType, Node> EoType;

  eoSubtreeXOver( unsigned _max_length)
    : eoQuadOp<EoType>(), max_length(_max_length) {};

  virtual string className() const { return "eoSubtreeXOver"; };

  /// Dtor
  virtual ~eoSubtreeXOver () {};

  void operator()(EoType & _eo1, EoType & _eo2 )
  {
	  int i = rng.random(_eo1.size());
	  int j = rng.random(_eo2.size());

	  parse_tree<Node>::subtree tmp = _eo2[j];
	  _eo1[i] = _eo2[j]; // insert subtree
	  _eo2[j]=tmp;
	  	  
	  _eo1.pruneTree(max_length);
	  _eo2.pruneTree(max_length);
	  
	  _eo1.invalidate();
	  _eo2.invalidate();
  }

  unsigned max_length;
};

template<class FType, class Node>
class eoBranchMutation: public eoMonOp< eoParseTree<FType, Node> > 
{
public:

  typedef eoParseTree<FType, Node> EoType;

  eoBranchMutation(eoInit<EoType>& _init, unsigned _max_length)
    : eoMonOp<EoType>(), max_length(_max_length), initializer(_init) 
  {};

  virtual string className() const { return "eoBranchMutation"; };

  /// Dtor
  virtual ~eoBranchMutation() {};

  void operator()(EoType& _eo1 ) 
  {
	  int i = rng.random(_eo1.size());
      
      EoType eo2;
      initializer(eo2);

	  int j = rng.random(eo2.size());

	  _eo1[i] = eo2[j]; // insert subtree
	  	  
	  _eo1.pruneTree(max_length);
	  	  
	  _eo1.invalidate();
  }

private :

  unsigned max_length;
  eoInit<EoType>& initializer; 
};


#endif
