
/* 
	DESCRIPTION:


The class 'Sym' in this package provides a reference counted, hashed tree structure that can be used in genetic programming.
The hash table behind the scenes makes sure that every subtree in the application is stored only once.
This has a couple of advantages:

o Memory: all subtrees are stored only once
o Comparison: comparison for equality for two subtrees boils down to a pointer comparison 
o Overview: by accessing the hashtable, you get an instant overview of the state of the population


The disadvantage of this method is the constant time overhead for computing hashes. In practice,
it seems to be fast enough.


===== How to Use this =========

In essence, the Sym data structure contains two important pieces of data,
the 'token' (of type token_t = int) and the children, a vector of Sym (called SymVec).
The token should contain all information to be able to figure out which 
function/terminal is represented by the node in the tree. By retrieving this token value
and the SymVec it is possible to write recursive traversal routines for evaluation, printing,
etc.

*/

#include <iostream>
#include "Sym.h"

using namespace std;


/* 
 * Suppose token value '0' designates our terminal, and token value '1' designates a binary function.
 * Later on a ternary function will be used as well, designated with token value '2'
 * The function below will create a tree of size three
*/
Sym test1() {
    
    SymVec children; 
    children.push_back( Sym(0) ); // push_back is a member from std::vector, SymVec is derived from std::vector
    children.push_back( Sym(0) );

    Sym tree = Sym(token_t(1), children); // creates the tree

    /* Done, now print some information about the node */

    cout << "Size      =  " << tree.size() << endl;     // prints 3
    cout << "Depth     = " << tree.depth() << endl;    // prints 2
    cout << "Refcount  = " << tree.refcount() << endl; // prints 1

    Sym tree2 = tree; // make a copy (this only changes refcount)

    cout << "Refcount now = " << tree.refcount() << endl; // print 2
    
    return tree; // tree2 will be deleted and reference count returns to 1
}

/* To actually use the tree, evaluate it, the following simple recursive function
 * can be used
*/

int eval(const Sym& sym) {
    if (sym.token() == 0) { // it's a terminal in this example
	return 1;
    }
    // else it's the function
    const SymVec& children = sym.args(); // get the children out, children.size() is the arity

    // let's assume that we've also got a ternary function designated by token '2'
    
    if (sym.token() == token_t(1))
	return eval(children[0]) + eval(children[1]); // evaluate

    return eval(children[0]) + eval(children[1]) * eval(children[2]); // a ternary function
}

/* Note that you simply use the stored token that was defined above. Simply checking the size of SymVec in
 * this particular example could have sufficed, but it's instructive to use the tokens.
 * 
 * And to test this: 
*/

void test_eval() {
    
    Sym tree = test1();

    cout << "Evaluating tree1 returns " << eval(tree) << endl;
}

/* Writing initialization functions.
 *
 * As the Sym class is recursive in nature, initialization can simply be done using 
 * recursive routines as above. As an example, the following code does 'full' initialization.
 */

Sym init_full(int depth_left) {
    if (depth_left == 0) return Sym(0); // create terminal
    // else create either a binary or a ternary function
    
    depth_left--;
    
    if (rand() % 2 == 0) { // create binary
	SymVec vec(2);
	vec[0] = init_full(depth_left);
	vec[1] = init_full(depth_left);

	return Sym(token_t(1), vec);
	
    } else { // create ternary tree
	SymVec vec(3);
	vec[0] = init_full(depth_left);
	vec[1] = init_full(depth_left);
	vec[2] = init_full(depth_left);

	return Sym(token_t(2), vec); // token value 2 designates a ternary now, even though the arity can simply be read from the size of the 'SymVec'
    }
    
}


/* Examining the hash table.
 *
 * The hash table is a static member of the Sym class, but can be obtained and inspected
 * at any point during the run. The hash table follows the SGI implementation of hashmap (and effectively
 * uses it in gcc). An example:
 */

void inspect_hashtable() {
    SymMap& dag = Sym::get_dag(); // get the hashmap
    unsigned i = 0;
    for (SymMap::iterator it = dag.begin(); it != dag.end(); ++it) {
	Sym node(it); // initialize a 'sym' with the iterator 

	cout << "Node " << i++ << " size " << node.size();
	cout << " refcount " << node.refcount()-1; // -1: note that by creating the Sym above the refcount is increased
	cout << " depth " << node.depth();
	cout << '\n';
    }
    
}

/* The above code effectively examines all distinct subtrees in use in the application and prints some stats for the node */

/* Manipulating trees 
 *
 * The Sym class is set up in such a way that you cannot change a Sym, so how do you perform crossover and mutation?
 *
 * Simple, you create new syms. The Sym class supports two functions to make this easier: 'get_subtree' and 'insert_subtree'.
 * These traverse the tree by index, where 0 designates the root and other values are indexed depth first.
 */

Sym subtree_xover(Sym a, Sym b) {
    
    Sym to_insert = get_subtree(a,  rand() % a.size() ); // select random subtree, will crash if too high a value is given
    
    /* 'insert' it into b. This will not really insert, it will however create a new sym,
     * equal to 'b' but with a's subtree inserted at the designated spot. */
    return insert_subtree(b, rand() % b.size(), to_insert); 
    
}

/* Tying it together, we can create a simple genetic programming system. Mutation is not implemented here,
 * but would be easy enough to add by using recursion and/or 'set'. */

void run_gp() {
    
    int ngens = 50;
    int popsize = 1000;

    cout << "Starting running " << popsize << " individuals for " << ngens << " generations." << endl;
    
    vector<Sym> pop(popsize); 
    
    // init population
    for (unsigned i = 0; i < pop.size(); ++i) {
	pop[i] = init_full(5); 
    }
    
    double best = 0.0;
    
    // do a very simple steady state tournament 
    for (unsigned gen = 0; gen < ngens * pop.size(); ++gen) {
	int sel1 = rand()% pop.size();
	int sel2 = rand() % pop.size();
	int sel3 = rand() % pop.size();

	double ev1 = eval(pop[sel1]);
	double ev3 = eval(pop[sel3]);
	
	double bst = max(ev1,ev3);
	if (bst > best) {
	    best = bst;
	}
	
	if (ev3 > ev1) {
	    sel1 = sel3; // selection pressure
	}

	Sym child = subtree_xover(pop[sel1], pop[sel2]);
	
	// Check for uniqueness
	if (child.refcount() == 1) pop[ rand() % pop.size() ] = child;
    }
    
    // and at the end:
    
    inspect_hashtable();

    // and also count number of nodes in the population
    int sz = 0;
    for (unsigned i = 0; i < pop.size(); ++i) { sz += pop[i].size(); }
    cout << "Number of distinct nodes " << Sym::get_dag().size() << endl;
    cout << "Nodes in population      " << sz << endl;
    cout << "ratio                    " << double(Sym::get_dag().size())/sz << endl;
    cout << "Best fitness	      " << best << endl;
    
}

/* One extra mechanism is supported to add annotations to nodes. Something derived from
 * 'UniqueNodeStats' can be used to attach new information to nodes. For this to function,
 * we need to supply a 'factory' function that creates these node-stats; attach this function to the 
 * Sym class, so that it gets called whenever a new node is created. The constructors of the Sym class
 * take care of this.
 *
 * IMPORTANT: 
 *	in a realistic application, the factory function needs to be set BEFORE any Syms are created
 *	Mixing Syms creating with and without the factory can lead to unexpected results    
 *
 * First we derive some structure from UniqueNodeStats: */

struct MyNodeStats : public UniqueNodeStats {
    
    int sumsize;
    
    ~MyNodeStats() { cout << "MyNodeStats::~MyNodeStats, sumsize = " << sumsize << endl; }
};

/* then define the factory function. It will get a Sym, which is just created.  */
UniqueNodeStats* create_stats(const Sym& sym) {
    MyNodeStats* stats = new MyNodeStats; // Sym will take care of memory management
    
    int sumsize = sym.size();
    for (unsigned i = 0; i < sym.args().size(); ++i) {
	// retrieve the extra node stats of the child
	UniqueNodeStats* unique_stats = sym.args()[i].extra_stats(); // extra_stats retrieves the stats
	MyNodeStats* child_stats = static_cast<MyNodeStats*>(unique_stats); // cast it to the right struct
	sumsize += child_stats->sumsize;
    }
    
    stats->sumsize = sumsize;
    return stats; // now it will get attached to the node and deleted when its reference count goes to zero
}

void test_node_stats() {
    
    if (Sym::get_dag().size() != 0) {
	cerr << "Cannot mix nodes with and without factory functions" << endl;
	exit(1);
    }
    
    /* Very Important: attach the factory function to the Sym class */
    Sym::set_factory_function(create_stats);

    Sym tree = init_full(5); // create a tree
    
    // get extra node stats out
    MyNodeStats* stats = static_cast<MyNodeStats*>( tree.extra_stats() );

    cout << "Size = " << tree.size() << " SumSize = " << stats->sumsize << endl;
    
    Sym::clear_factory_function(); // reset
}


/* And run the code above */

int main() {
    srand(time(0));
    cout << "********** TEST EVALUATION **************\n";
    test_eval();
    cout << "********** TEST ALGORITHM ***************\n";
    run_gp();

    cout << "********** TEST FACTORY  ****************\n";
    test_node_stats(); // can work because there are no live nodes

}

/* ********** Member function reference: ********************
 *
 * Sym()	    The default constructor will create an undefined node (no token and no children), check for empty() to see if a node is undefined
 * 
 * Sym(token_t)	    Create a terminal
 *
 * Sym(token_t, const SymVec&)
 *		    Create a node with token and SymVec as the children
 * 
 * Sym(SymIterator it)
 *		    Create a sym from an iterator (taken from the hashtable directly, or from Sym::iterator)
 *
 * dtor, copy-ctor and assignment
 *
 * UniqueNodeStats* extra_stats()    
 *		    Returns an UniqueNodeStats pointer (= 0 if no factory is defined)
 * 
 * 
 * int hashcode()   returns the hashcode for the node
 * 
 * int refcount()   returns the reference count for the node
 * 
 * bool operator==  checks for equality (note that this is a pointer compare, really really fast)
 * 
 * bool empty()	    returns whether the node is undefined, i.e. created through the default ctor 
 * 
 * int arity()	    shorthand for sym.args().size()
 * 
 * token_t token()  return identifying token for the node
 * 
 * const SymVec& args()
 *		    returns the children of the node (in a vector<Sym>)
 *		    
 * unsigned size()  returns the size, i.e., number of nodes
 * 
 * unsigned depth() returns the depth
 * 
 * iterator()       returns the pointer to the node in the hashtable
 *
 * 
 ********** Static functions: ********************
 *
 *
 * 
 * SymMap& get_dag()	returns the hash table containing all nodes. This should only be used for inspection,
 *			even though the dag itself is not const. This to enable the use of the ctor Sym(SymIterator) to inspect
 *			using the Sym interface (rather than the hash table interface). This does allow you to make destructive
 *			changes to the class, so use with care
 *
 * set_factory_function( UniqueNodeStats (*)(const Sym&) )
 *			Set the factory function
 *
 *  clear_factory_function()	
 *			Clears the factory function, allocated UniqueNodeStats will still be deleted, but no new ones will be created.
 *
 ********** Utility Functions ******************** 
 * 
 * Sym get_subtree(const Sym& org, int i)
 *			Retreive the i-th subtree from the Sym. Standard depth first ordering, where root has index 0 and the
 *			rightmost terminal has index sym.size()-1
 *
 * Sym insert_subtree(const Sym& org, int i, const Sym& subtree)
 *			Returns a Sym that is equal to 'org', for which the i-th subtree (same ordering as get_subtree) is replaced
 *			by the third argument subtree.
 * 
 * Sym next(const Sym&)
 *			Returns the successor of the argument sym from the hashtable with wrap around. This is implemented just because
 *			it can be done. It may be an interesting way to mutate...
 * 
 * */


