#ifndef EO_PARSE_TREE_H
#define EO_PARSE_TREE_H

#include <list>

#include <EO.h>
#include <eoOp.h>
#include <eoInserter.h>
#include <eoIndiSelector.h>
#include <gp/parse_tree.h>
#include <eoInit.h>

using namespace gp_parse_tree;
using namespace std;

template <class FType, class Node>
class eoParseTree : public EO<FType>, public parse_tree<Node>
{
public :
    
    typedef typename parse_tree<Node>::subtree Type;

    eoParseTree(void) : EO<FType>(), parse_tree<Node>() {}
    eoParseTree(unsigned _size, eoRnd<Type>& _rnd) 
        : EO<FType>(), parse_tree<Node>(_rnd()) 
    {
        pruneTree(_size);
    }
    eoParseTree(eoRnd<Type>& _rnd) 
        : EO<FType>(), parse_tree<Node>(_rnd()) 
    {}

    virtual void pruneTree(unsigned _size)
    {
        if (_size < 1)
            return;

        if (size() > _size)
        {
            Type* sub = &operator[](size() - 2); // prune tree

            while (sub->size() > _size)
            {
                sub = &sub->operator[](0);
            }

            back() = *sub;
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
            eoRnd<EoType::Type>(),
			max_depth(_max_depth),
			initializor(_initializor),
			grow(_grow) 
    {}

	virtual string className() const { return "eoDepthInitializer"; };

    void operator()(EoType& _tree)
	{
        list<Node> sequence;
        
        generate(sequence, max_depth);
        
        _tree = parse_tree<Node>(sequence.begin(), sequence.end());
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
class eoSubtreeXOver: public eoGeneralOp< eoParseTree<FType, Node> > {
public:

  typedef eoParseTree<FType, Node> EoType;

  eoSubtreeXOver( unsigned _max_length)
    : eoGeneralOp<EoType>(), max_length(_max_length) {};

  virtual string className() const { return "eoSubtreeXOver"; };

  /// Dtor
  virtual ~eoSubtreeXOver () {};

  void operator()(eoIndiSelector<EoType>& _source, eoInserter<EoType>& _sink ) const
  {
      EoType eo1 = _source.select();
      const EoType& eo2 = _source.select();

	  int i = rng.random(eo1.size());
	  int j = rng.random(eo2.size());

	  eo1[i] = eo2[j]; // insert subtree
	  	  
	  eo1.pruneTree(max_length);
	  
	  eo1.invalidate();
      _sink.insert(eo1);
  }

  unsigned max_length;
};

template<class FType, class Node>
class eoBranchMutation: public eoGeneralOp< eoParseTree<FType, Node> > 
{
public:

  typedef eoParseTree<FType, Node> EoType;

  eoBranchMutation(eoRnd<EoType::Type>& _init, unsigned _max_length)
    : eoGeneralOp<EoType>(), max_length(_max_length), initializer(_init) 
  {};

  virtual string className() const { return "eoBranchMutation"; };

  /// Dtor
  virtual ~eoBranchMutation() {};

  void operator()(eoIndiSelector<EoType>& _source, eoInserter<EoType>& _sink ) const
  {
      EoType eo1 = _source.select();
	  int i = rng.random(eo1.size());
      
      EoType eo2(eo1[i].size(), initializer); // create random other to cross with

	  eo1[i] = eo2.back(); // insert subtree
	  	  
	  eo1.pruneTree(max_length);

	  eo1.invalidate();
      _sink.insert(eo1);
  }

private :

  unsigned max_length;
  eoRnd<EoType::Type>& initializer; 
};


#endif
