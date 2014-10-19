// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoGenOp.cpp
// (c) Maarten Keijzer and Marc Schoenauer, 2001
/*
    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Contact: mkeijzer@dhi.dk
	     Marc.Schoenauer@polytechnique.fr
 */
//-----------------------------------------------------------------------------

/**  test program for the general operator - millenium version!
 * uses dummy individuals
 */

#include <sstream>

#include <paradiseo/eo.h>
#include <paradiseo/eo/eoPopulator.h>
#include <paradiseo/eo/eoOpContainer.h>

struct Dummy : public EO<double>
{
    Dummy(std::string _s="") : s(_s) {}

    void printOn(std::ostream & _os) const
	{
	    EO<double>::printOn(_os);
	    _os << " - " << s ;
	}

    std::string s;
};

typedef Dummy EOT;

unsigned int pSize;	 // global to be used as marker in the fitness

// DEFINITIONS of the eoOps
class monop : public eoMonOp<EOT>
{
  public :
  monop(char * _sig){sig=_sig;}
    bool operator()(EOT& _eo)
    {
      _eo.s = sig + "(" + _eo.s + ")";
      _eo.fitness(_eo.fitness()+pSize);
      return false;
    }
  std::string className() const {return sig;}
    private:
    std::string sig;
};

class binop: public eoBinOp<EOT>
{
  public :
    bool operator()(EOT& _eo1, const EOT& _eo2)
    {
      _eo1.s = "bin(" + _eo1.s + "," + _eo2.s + ")";
      double f= (_eo1.fitness()+_eo2.fitness()) * pSize;
      _eo1.fitness(_eo1.fitness()+f);
      return false;
    }
  std::string className() const {return "binop";}
};

class quadop: public eoQuadOp<EOT>
{
  public :
  std::string className() const {return "quadop";}
    bool operator()(EOT& a, EOT& b)
    {
      EOT oi = a;
      EOT oj = b;

      a.s = "quad1(" + oi.s + "," + oj.s + ")";
      b.s = "quad2(" + oj.s + "," + oi.s + ")";
      double f= (a.fitness()+b.fitness()+2*pSize) * pSize;
      a.fitness(a.fitness()+f);
      b.fitness(b.fitness()+f);
      return false;
    }
};
// an eoQuadOp that does nothing
class quadClone: public eoQuadOp<EOT>
{
  public :
  std::string className() const {return "quadclone";}
    bool operator()(EOT& , EOT& ) {return false;}
};

// User defined General Operator... adapted from Marc's example

class one2threeOp : public eoGenOp<EOT> // :-)
{
  public:
    unsigned max_production(void) { return 3; }

    void apply(eoPopulator<EOT>& _plop)
    {
      EOT& eo = *_plop; // select the guy

      ++_plop; // advance
      _plop.insert("v(" + eo.s + ", 1)");
      ++_plop;
      _plop.insert("v(" + eo.s + ", 2)");
      eo.s  =  "v(" + eo.s + ", 0)"; // only now change the thing
      // oh right, and invalidate fitnesses
    }
  virtual std::string className() const {return "one2threeOp";}
};


class two2oneOp : public eoGenOp<EOT> // :-)
{
  public:
    unsigned max_production(void) { return 1; }

    void apply(eoPopulator<EOT>& _plop)
    {
      EOT& eo = *_plop; // select the guy
      const EOT& eo2 = _plop.select();
      eo.s  =  "221(" + eo.s + ", " + eo2.s + ")";
      // oh right, and invalidate fitnesses
    }
  virtual std::string className() const {return "two2oneOp";}
};

class three2threeOp : public eoGenOp<EOT> // :-)
{
  public:
    unsigned max_production(void) { return 3; }

    void apply(eoPopulator<EOT>& _plop)
    {
      EOT& eo1 = *_plop; // select 1st guy
      EOT& eo2 = *++_plop; // select 2nd guy
      EOT& eo3 = *++_plop; // select 3rd guy
      EOT a = eo1;
      EOT b = eo2;
      EOT c = eo3;
      std::cout << "les selectionnes: a=" << a << " et b=" << b << " et c=" << c << std::endl;
      eo1.s  =  "323-1(" + a.s + ", " + b.s + ", " + c.s + ")";
      eo2.s  =  "323-2(" + a.s + ", " + b.s + ", " + c.s + ")";
      eo3.s  =  "323-3(" + a.s + ", " + b.s + ", " + c.s + ")";
      // oh right, and invalidate fitnesses
      std::cout << "les enfants: a=" << eo1 << " et b=" << eo2 << " et c=" << eo3 << std::endl;
    }
  virtual std::string className() const {return "three2threeOp";}
};


// dummy intialization. Re-init if no pSize, resize first if pSize
void init(eoPop<Dummy> & _pop, unsigned _pSize)
{
  if (_pSize)
    {
      _pop.resize(_pSize);
    }
  else
    {
      throw std::runtime_error("init pop with 0 size");
    }
  for (unsigned i=0; i<_pSize; i++)
    {
	std::ostringstream os;
	os << i;
	_pop[i] = Dummy(os.str());
	_pop[i].fitness(i);
    }
}

// ok, now for the real work
int the_main(int argc, char **argv)
{
    eoParser parser(argc, argv);
    eoValueParam<unsigned int> parentSizeParam(
	parser.createParam(unsigned(10), "parentSize", "Parent size",'P'));
    pSize = parentSizeParam.value(); // global variable

    eoValueParam<uint32_t> seedParam(time(0), "seed", "Random number seed", 'S');
    parser.processParam( seedParam );
    eo::rng.reseed(seedParam.value());

   // do the following AFTER ALL PARAMETERS HAVE BEEN PROCESSED
   // i.e. in case you need parameters somewhere else, postpone these
    if (parser.userNeedsHelp())
      {
	parser.printHelp(std::cout);
	exit(1);
      }

  ////////////////////////////////// define operators
  monop mon((char*)"mon1");
  monop clone((char*)"clone");
  binop bin;
  quadop quad;
  quadClone quadclone;

  // our own operator
  one2threeOp o2t;
  two2oneOp t2o;
  three2threeOp t2t;


  // a selector
  eoDetTournamentSelect<EOT> select;
  // and a recognizable selector for testing the inbedded selector mechanism
  eoBestSelect<EOT> selectBest;

  // proportional selection between quad and bin
  // so we either do a quad or a bin
  eoProportionalOp<EOT> pOp;
  pOp.add(quad, 0.1);
  pOp.add(bin, 0.1);

  // sequential selection between pOp and mon
  eoSequentialOp<EOT> sOp;
  sOp.add(pOp, 0.9);
  sOp.add(mon, 0.1);

  // with one2three op
  eoSequentialOp<EOT> sOp2;
  sOp2.add(o2t, 1);
  //  sOp2.add(quad, 1);

  // with three2three op
  eoSequentialOp<EOT> sOp3;
  sOp3.add(t2t, 1);

//   eoSequentialOp<EOT> sOp3;
//   sOp3.add(t2o, 1);
//   sOp3.add(bin, 1);
//   sOp3.add(quad, 1);
  // try adding quads and bins to see what results you'll get

  // now a sequential selection that is a simple "addition"
  eoSequentialOp<EOT> sOpQuadPlusMon;
  sOpQuadPlusMon.add(quad, 1);
  sOpQuadPlusMon.add(mon, 1);

  // this corresponds
  eoProportionalOp<EOT> pOpSAGLike;
  pOpSAGLike.add(sOpQuadPlusMon, 0.24);
  pOpSAGLike.add(quad, 0.56);
  pOpSAGLike.add(mon, 0.06);
  pOpSAGLike.add(clone, 0.14);

  // init
  eoPop<EOT> pop;
  eoPop<EOT> offspring;

  init(pop, pSize);
// sort pop so seqPopulator is identical to SelectPopulator(SequentialSelect)
  pop.sort();
  std::cout << "Population initiale" << std::endl << pop << std::endl;

  // To simulate SGA: first a prop between quadOp and quadClone
  eoProportionalOp<EOT> pSGAOp;
  pSGAOp.add(quad, 0.8);
  pSGAOp.add(quadclone, 0.2);
  // sequential selection between pSGAOp and mon
  eoSequentialOp<EOT> virtualSGA;
  virtualSGA.add(pSGAOp, 1.0);
  virtualSGA.add(mon, 0.3);

  eoSeqPopulator<EOT> popit(pop, offspring);  // no selection, a copy of pop

  // until we filled a new population
  try
    {
     while (offspring.size() < pop.size())
	   {
	     virtualSGA(popit);
	     std::cout << "SeqPopulator boucle et incremente\n";
       ++popit;
	   }
    }
  catch(eoPopulator<EOT>::OutOfIndividuals&)
    {
      std::cout << "Warning: not enough individuals to handle\n";
    }


  std::swap(pop, offspring);
  offspring.clear();

  // ok, now print
  std::cout << "Apres virtualSGA \n" << pop << std::endl;
  init(pop, pSize);

  std::cout << "=========================================================\n";
  std::cout << "Now the eoSelectPopulator version !" << std::endl;

  eoSequentialSelect<EOT> seqSelect;
  //   select.init(); should be sorted out: is it the setup method???
  eoSelectivePopulator<EOT> it_step3(pop, offspring, seqSelect);

  while (offspring.size() < 2*pop.size())
  {
    virtualSGA(it_step3);
	     std::cout << "SelectPopulator boucle et incremente\n";
    ++it_step3;
  }

  std::swap(pop, offspring);
  offspring.clear();

    // ok, now print
  std::cout << "Apres SGA-like eoSelectivePopulator\n" << pop << std::endl;

  std::cout << "=========================================================\n";
  std::cout << "Now the pure addition !" << std::endl;

  init(pop, pSize);
  eoSelectivePopulator<EOT> it_step4(pop, offspring, seqSelect);
  while (offspring.size() < 2*pop.size())
  {
    sOpQuadPlusMon(it_step4);
    ++it_step4;
  }

  std::swap(pop, offspring);
  offspring.clear();

    // ok, now print
  std::cout << "Apres Quad+Mon ds un eoSelectivePopulator\n" << pop << std::endl;

  // On teste 1->3
  init(pop, pSize);
  eoSelectivePopulator<EOT> it_step5(pop, offspring, seqSelect);
  while (offspring.size() < 2*pop.size())
  {
    sOp2(it_step5);
    ++it_step5;
  }

  std::swap(pop, offspring);
  offspring.clear();

    // ok, now print
  std::cout << "Apres 1->3 seul ds un eoSelectivePopulator\n" << pop << std::endl;

  // On teste 3->3
  init(pop, pSize);
  eoSelectivePopulator<EOT> it_step6(pop, offspring, seqSelect);
  while (offspring.size() < 2*pop.size())
  {
    sOp3(it_step6);
    ++it_step6;
  }

  std::swap(pop, offspring);
  offspring.clear();

    // ok, now print
  std::cout << "Apres 3->3 seul ds un eoSelectivePopulator\n" << pop << std::endl;


  return 1;
}

int main(int argc, char **argv)
{
    try
    {
	the_main(argc, argv);
    }
    catch(std::exception& e)
    {
	std::cout << "Exception: " << e.what() << std::endl;
    }

}

/*
If you want to build an SGA, you will need a copying quad op:

class quadclone : ...
{
  operator(EOT& a, EOT& b)
  {
    // do nothing
  }

}

Then the SGA operator will look like:

quadop quad;
guadclone clone;

ProportionalGenOp pOp;
pOp.add(quad, 0.8);
pOp.add(clone, 0.2); // so 80% xover rate

SequentialGenOp sOp;
sOp.add(pOp, 1,0); // always try a xover (clone 20%)
sOp.add(mut, 0.1); // low mutation rate

will result in an algorithm with:

p_xover = 0.8
p_mut   = 0.1;

p_reproduction = 0.2 * 0.9 = 0.18

this does not add up to 1 because xover and mutation can be applied to a single indi

So what do you think?

*/
