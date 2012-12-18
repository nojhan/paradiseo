#include <map> // for pair
#include <iostream>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <math.h> // for exp

using namespace std;

/* fitness_traits.h */

// default traits: defaults to a double that needs to be maximized
template <class T = double>
struct fitness_traits
{
  // Needs mapping can be used to figure out whether you need to do fitness scaling (or not)
  const static bool needs_mapping               = false;

  // storage_type: what to store next to the genotype
  typedef T storage_type;

  // performance_type: what the eoEvalFunc calculates
  typedef T performance_type;

  // worth_type: what the scaling function does
  typedef T worth_type;

  // access_performance: how to get from what is stored to a mutable performance
  static performance_type& access_performance(storage_type& a) { return a; }

  // access_worth: how to get from what is stored to a mutable worth
  static worth_type&       access_worth(storage_type& a)       { return a; }

  // get_performance: from storage_type to a performance figure
  static performance_type  get_performance(storage_type a) { return a; }

  // get_worth: from storage_type to a worth figure
  static worth_type        get_worth(storage_type a)       { return a; }

  // get the fitness out of the individual
  template <class EOT>
  static worth_type    get_fitness(const EOT& _eo)       { return _eo.performance(); }

  // compare the two individuals
  template <class EOT>
  static bool is_better(const EOT& _eo1, const EOT& _eo2)
  {
    return _eo1.performance() > _eo2.performance();
  }
};

struct minimization {};
struct maximization {};

struct fitness_traits<minimization> : public fitness_traits<double>
{
  // for minimization, invert the is_better
  template <class EOT>
  static bool is_better(const EOT& _eo1, const EOT& _eo2)
  {
    return _eo1.performance() < _eo2.performance();
  }
};

// for maximization, just take the default behaviour
struct fitness_traits<maximization> : public fitness_traits<double> {};

// forward declaration
//template <class EOT> class eoPop;
//template <class Fitness, class Traits> class EO;

// unfortunately, partial template specialization is not approved by Microsoft (though ANSI says it's ok)
// Probably need some macro-magic to make this work (MicroSoft == MacroHard)
// A pair class: first == performance, second == worth, redefine all types, data and functions
template <class Performance, class Worth>
struct fitness_traits< pair<Performance, Worth> >
{
  typedef pair<Performance, Worth> storage_type;
  typedef Performance performance_type;
  typedef Worth       worth_type;

  const static bool needs_mapping = true;

  static performance_type& access_performance(storage_type& a)     { return a.first; }
  static worth_type&       access_worth(storage_type& a)           { return a.second; }

  static performance_type get_performance(const storage_type& a)     { return a.first; }
  static worth_type       get_worth(const storage_type& a)           { return a.second; }

  // This function calls _eo.worth() which in turn checks the fitness flag and calls get_worth above
  // The compiler should be able to inline all these calls and come up with a very compact solution
  template <class EOT>
  static worth_type    get_fitness(const EOT& _eo) { return _eo.worth(); }

  template <class EOT>
  static bool is_better(const EOT& _eo1, const EOT& _eo2)
  {
    return _eo1.worth() > _eo2.worth();
  }
};

/* end fitness_traits.h */

/* EO.h

The Fitness template argument is there for backward compatibility reasons

*/

template <class Fitness, class Traits = fitness_traits<Fitness> >
class EO
{
public :

  typedef Traits                            fitness_traits;
  typedef typename Traits::storage_type     storage_type;
  typedef typename Traits::performance_type performance_type;
  typedef typename Traits::worth_type       worth_type;

  EO() : valid_performance(false), valid_worth(false), rep_fitness() {}

  // for backwards compatibility
  void fitness(performance_type perf)
  {
    performance(perf);
  }

  void performance(performance_type perf)
  {
    valid_performance = true;
    Traits::access_performance(rep_fitness) = perf;
  }

  performance_type performance(void) const
  {
    if(!valid_performance) throw runtime_error("no performance");
    return Traits::get_performance(rep_fitness);
  }

  void worth(worth_type worth)
  {
    valid_worth = true;
    Traits::access_worth(rep_fitness) = worth;
  }

  worth_type worth(void) const
  {
    if(!valid_worth)  throw runtime_error("no worth");
    if(!Traits::needs_mapping)  throw runtime_error("no mapping");
    return Traits::get_worth(rep_fitness);
  }

  worth_type fitness(void) const
  {
    return Traits::get_fitness(*this);
  }

  void invalidate(void)
  {
    valid_performance = false;
    valid_worth = false;
  }

  void invalidate_worth(void)
  {
    valid_worth = false;
  }

  bool operator<(const EO<Fitness, Traits>& other) const
  {
    return !Traits::is_better(other, *this);
  }

  bool operator>(const EO<Fitness, Traits>& other) const
  {
    return Traits::is_better(other, *this);
  }

  private :

  bool valid_performance;
  bool valid_worth;
  storage_type rep_fitness;
};

/* end EO.h */

/* eoPerf2Worth.h */

// get the name known
template <class EOT> class eoPop;

template <class EOT>
void exponential_scaling(eoPop<EOT>& _pop)
{
    for (unsigned i = 0; i < _pop.size(); ++i)
    { // change minimimization into maximization
      _pop[i].worth(exp(-_pop[i].performance()));
    }
}

template <class EOT>
class eoPerf2Worth /* : public eoUF<eoPop<EOT>&, void> */
{
public :
  virtual void operator()(eoPop<EOT>& _pop)
  {
    return exponential_scaling(_pop);
  }
};

/* end eoPerf2Worth.h */


/* eoPop.h */

template <class EOT>
class eoPop : public vector<EOT>
{
public :

  typedef typename EOT::fitness_traits fitness_traits;

  eoPop(void) : p2w(0) {}

  void sort()
  {
    scale(); // get the worths up to date

    std::sort(begin(), end(), greater<EOT>());
  }

  void scale()
  {
    if (p2w)
    {
      if (!fitness_traits::needs_mapping)
      {
	throw runtime_error("eoPop: no scaling needed, yet a scaling function is defined");
      }

      (*p2w)(*this);
    }
    else if (fitness_traits::needs_mapping)
    {
      throw runtime_error("eoPop: no scaling function attached to the population, while one was certainly called for");
    }
  }

  void setPerf2Worth(eoPerf2Worth<EOT>& _p2w)
  {
    p2w = &_p2w;
  }

  void setPerf2Worth(eoPerf2Worth<EOT>* _p2w)
  {
    p2w = _p2w;
  }

  eoPerf2Worth<EOT>* getPerf2Worth() { return p2w; }

  void swap(eoPop<EOT>& other)
  {
    vector<EOT>::swap(other);
    eoPerf2Worth<EOT>* tmp = p2w;
    p2w = other.p2w;
    other.p2w = tmp;
  }

private :

  // a pointer as it can be emtpy
  eoPerf2Worth<EOT>* p2w;
};

// need this one to be able to swap the members as well...
template <class EOT>
void swap(eoPop<EOT>& _p1, eoPop<EOT>& _p2)
{
  _p1.swap(_p2);
}

/* end eoPop.h */

/* main and test */

template <class EOT>
void algo(eoPop<EOT>& _pop)
{
  eoPop<EOT> offspring;                          // how to get the scaling info into this guy??
  offspring.setPerf2Worth(_pop.getPerf2Worth()); // like this!

  std::copy(_pop.begin(), _pop.end(), back_inserter(offspring));

  offspring.sort(); // should call scale

  swap(_pop, offspring);
}

void minimization_test()
{
  typedef EO<minimization> eo_type;

  eo_type eo1;
  eo_type eo2;

  eo1.performance(1.0);
  eo2.performance(2.0);

  std::cout << "With minimizing fitness" << std::endl;
  std::cout << eo1.fitness() << " < " << eo2.fitness() << " returns " << (eo1 < eo2) << std::endl;
  std::cout << eo2.fitness() << " < " << eo1.fitness() << " returns " << (eo2 < eo1) << std::endl;
}

void the_main()
{
  typedef EO<double> simple_eo;
  typedef EO<pair<double, double> > scaled_eo;

  simple_eo eo1;
  simple_eo eo3;

/* First test some simple comparisons */

  eo1.fitness(10); // could also use performance()
  eo3.fitness(5);

  std::cout << eo1.fitness() << std::endl;
  std::cout << eo3.fitness() << std::endl;

  std::cout << "eo1 < eo3 = " << (eo1 < eo3) << std::endl;


  scaled_eo eo2;
  scaled_eo eo4;
  eo2.performance(10);
  eo4.performance(8);

/* Now test if the worth gets accessed and if the flag protects it */

  try
  {
    std::cout << eo2.fitness() << std::endl;
    std::cout << "did not throw" << std::endl;
    assert(false); // should throw
  }
  catch(std::exception& e)
  {
    std::cout << "Fitness threw exception, as it should" << std::endl;
    std::cout << e.what() << std::endl;
  }

/* Set the worth and all is well (this is normally done by some perf2worth functor */

  eo2.worth(3);
  eo4.worth(5);

  std::cout << "with maximization " << std::endl;
  std::cout << eo2.fitness() << std::endl;
  std::cout << eo4.fitness() << std::endl;
  std::cout << eo2.fitness() << " < " << eo4.fitness() << " returns " << (eo2 < eo4) << std::endl;

/* Test the minimization of fitness */
  minimization_test();


/* Populations */

// test pop without scaling, should have no overhead save for a single empty pointer in pop
  eoPop<simple_eo> pop0;
  pop0.resize(1);
  pop0[0].fitness(1);

  algo(pop0);

  std::cout << pop0[0].fitness() << std::endl;

  assert(pop0[0].fitness() == 1);

/* test pop with scaling */

  eoPerf2Worth<scaled_eo> perf2worth;
  eoPop<scaled_eo> pop1;

  pop1.resize(1);

  pop1[0].fitness(1.0); // emulate evaluation

    // at this point getting the fitness should throw
  try
  {
    std::cout << pop1[0].fitness() << std::endl;
    std::cout << "did not throw" << std::endl;
    assert(false); // should throw
  }
  catch(std::exception& e)
  {
    std::cout << "Fitness threw exception, as it should" << std::endl;
    std::cout << e.what() << std::endl;
  }

  // at this point trying to scale should throw
  try
  {
    algo(pop1); // should complain that it cannot scale
    assert(false); // so it would never get here
  }
  catch(std::exception& e)
  { // but rather ends here
    std::cout << e.what() << std::endl;
  }

  // ok, now set the scaling
  pop1.setPerf2Worth(perf2worth);

  algo(pop1);

  std::cout << "the fitness has been transformed from " << pop1[0].performance() << " to exp(-1) = " << pop1[0].fitness() << std::endl;
}

int main()
{
  try
  {
    the_main();
  }
  catch(std::exception& e)
  {
    std::cout << e.what() << std::endl;
  }
}
