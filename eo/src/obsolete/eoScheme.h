// eoScheme.h
// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoScheme.h 
// (c) GeNeura Team, 1998 - EEAAX 1999
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

   Contact: todos@geneura.ugr.es, http://geneura.ugr.es
 */
//-----------------------------------------------------------------------------

#ifndef _EOSCHEME_H
#define _EOSCHEME_H

using namespace std;

/**
@author Geneura Team -- EEAAX 99
@version 0.0
                   Evolution scheme


 It seems to me that some important conceptual object is missing
   (and God knows I hate turning everything into objects!

 So here is the evolution scheme: 

 regroups selection (nb of offspring to generate AND selection method
 and replacement (from parents + offspring)

 allows to include elitism, eugenism of the worst, ... more easily

 a generation is then simply an evolution scheme and some operators
 and a full algo is a generation + a termination condition

 this is mainly a container class
*/

//-----------------------------------------------------------------------------

#include <eoPopOps.h>
#include "eoStochTournament.h"
#include "eoDetTournament.h"
#include "eoLottery.h"
#include "eoUniformSelect.h"
#include "eoInclusion.h"
#include "eoESReplace.h"
#include "eoEPTournament.h"

/** eoScheme is a class that does more sophisticated evolution that eoEasyEA
 */
template<class EOT>
class eoScheme: public eoAlgo<EOT>{

 public:
  
  // Dtor
  virtual ~eoScheme() {};

  // copy ctor is impossible because of pointers to pure virual types.
  // any idea???? --- leave the default copy ctor -- JJ

  /** the Parser-based constructor
   these switch cases could be turned into as many subclasses 
   - but how do you return the subclass to the caller of the constructor???
  */
  eoScheme(Parser & parser) {
    // read the popsize 
    parser.AddTitle("Description of evolution");
    string Evol;
    string SelectString;
    // temporary
    float rate_offspring;
    
    try {
      Evol = parser.getString("-EE", "--evolution", "GGA", 
			      "Evolution scheme (GGA, SSGA, ESPlus, ESComma, EP, General)" );
      popsize = parser.getInt("-EP", "--population", "10", 
			      "Population size" );
    }
    catch (exception & e)
      {
	cout << e.what() << endl;
	parser.printHelp();
	exit(1);
      }
    // now the big switch
    if (! strcasecmp(Evol.c_str(), "GGA") ) {
      // GGA parameters: popsize, selection method (and its parameters) 
      nb_offspring = 0;
      rate_offspring = 1.0;	// generational replacement: #offspring=popsize
      try { 
	//      parser.AddTitle("GGA Parameters");
	SelectString = parser.getString("-ES", "--selection", "Tournament", 
					"Selection method (Roulette, tournament)" );
	if (! strcasecmp(SelectString.c_str(), "roulette") ) {
	  ptselect = new eoLottery<EOT> ();
	  ptselect_mate = new eoLottery<EOT> ();
	} 
	if (! strcasecmp(SelectString.c_str(), "tournament") ) {
	  float rate = parser.getFloat("-Et", "--TselectSize", "2", 
				       "Tournament size or rate" );
	  if (rate < 0.5)
	    throw out_of_range("Invalid tournament rate");
	  else if ( rate < 1 ) {     // binary stochastic tournament
	    ptselect = new eoStochTournament<EOT>(rate); 
	    ptselect_mate = new eoStochTournament<EOT>(rate); 
	  } else { // determeinistic tournament of size (int)rate
	    ptselect = new eoDetTournament<EOT>((int)rate);
	    ptselect_mate = new eoDetTournament<EOT>((int)rate);
	  }
	}
	// end choice of selection
	ptreplace = new eoInclusion<EOT>();
	// put here the choice of elitism
      } 
      catch (exception & e)
	{
	  cout << e.what() << endl;
	  parser.printHelp();
	  exit(1);
	}
    }  // end of GGA	
    
    // SSGA - standard, one offspring only at the moment
    if (! strcasecmp(Evol.c_str(), "SSGA") ) {
      // SSGA parameters: popsize, selection tournament size
      // the replacement is limited to repace_worst, though
      // it could be easy to add the anti-tournament replacement method
      nb_offspring = 1;
      // NOTE: of course it's a bit of a waste to use the standard loop
      // for one single offspring ...
      try { 
	//      parser.AddTitle("SSGA Parameters");
	float _rate = parser.getFloat("-ET", "--TSelectSize", "2", 
				      "Selection tournament size" );
	if ( _rate < 1 ) {     // binary stochastic tournament
	  ptselect = new eoStochTournament<EOT>(_rate); 
	  ptselect_mate = new eoStochTournament<EOT>(_rate); 
	} else { // determeinistic tournament of size (int)rate
	  ptselect = new eoDetTournament<EOT>((int)_rate);
	  ptselect_mate = new eoDetTournament<EOT>((int)_rate);
	}
      }
      catch (exception & e)
	{
	  cout << e.what() << endl;
	  parser.printHelp();
	  exit(1);
	}
      // end choice of selection
      ptreplace = new eoInclusion<EOT>();
      // put here the choice of elitism
    }  // end of SSGA	
    
    if (! strcasecmp(Evol.c_str(), "ESPlus") ) {
      // ES evolution parameters: lambda = _nb_offspring
      
      try { 
	//      parser.AddTitle("ES Scheme parameters");
	nb_offspring = parser.getInt("-EL", "--lambda", "50", 
				     "Lambda" );
	ptselect = new eoUniformSelect<EOT>();
	ptselect_mate = new eoUniformSelect<EOT>();
	ptreplace =  new eoESPlus<EOT>();
      } 
      catch (exception & e)
	{
	  cout << e.what() << endl;
	  parser.printHelp();
	  exit(1);
	}
    }  // end of ESPlus
    
    if (! strcasecmp(Evol.c_str(), "ESComma") ) {
      // ES evolution parameters: lambda = _nb_offspring
      
      try { 
	//      parser.AddTitle("ES Scheme parameters");
	nb_offspring = parser.getInt("-EL", "--lambda", "50", 
				     "Lambda" );
	ptselect =  new eoUniformSelect<EOT>();
	ptselect_mate =  new eoUniformSelect<EOT>();
	ptreplace = new eoESComma<EOT>();
      } 
      catch (exception & e)
	{
	  cout << e.what() << endl;
	  parser.printHelp();
	  exit(1);
	}
    }  // end of ESCOmma
    
    if (! strcasecmp(Evol.c_str(), "EP") ) {
      // EP evoltion scheme: only the EP-tournament replacement size is neede
      
      try { 
	//      parser.AddTitle("EP Scheme parameters");
	nb_offspring = popsize;
	ptselect =  new eoCopySelect<EOT>;		/* no selection */
	ptselect_mate =  new eoUniformSelect<EOT>();
				/* What, crossover in EP :-) */
	unsigned tsize = parser.getInt("-ET", "--TournamentSize", "6", 
				       "Size of stocahstic replacement tournament" );
	ptreplace = new eoEPTournament<EOT>(tsize);
      } 
      catch (exception & e)
	{
	  cout << e.what() << endl;
	  parser.printHelp();
	  exit(1);
	}
    }  // end of EP
    
    
    // everyting is read: now the consistency checks and other preliminary steps
    nb_offspring = (nb_offspring ? nb_offspring : 
		    (int) (rate_offspring * popsize) );
    if (!nb_offspring)
      nb_offspring = 1;		/* al least one offspring */
    
  }
  
  
  // accessors
  unsigned PopSize(){return popsize;}
  unsigned NbOffspring(){return nb_offspring ;}
  eoSelect<EOT>* PtSelect(){return ptselect;}
  eoSelect<EOT>* PtSelectMate(){return ptselect_mate;}
  eoMerge<EOT>* PtMerge() {return ptreplace;}
  // NOTE: need pointers otherwise you have many warnings when initializing  

  /** @name Methods from eoObject	*/
  //@{
  /** Inherited from eoObject. Returns the class name.
      @see eoObject
  */
  string className() const {return "eoScheme";};
  //@}
 private:
  unsigned popsize;		/* but should it be here ??? */
  unsigned nb_offspring;	/* to generate through selection+operators */

  // these are provisional for later use

  //  float rate_offspring;		/* or rate */
  //  unsigned nb_survive;		/* the best guys that are transmitted anyway */
  //  float rate_survive;		/* or rate */
  //  unsigned nb_die;	  /* the worst guys that do not enev enter selection */
  //  float rate_die;		/* or rate */

  eoSelect<EOT> *    ptselect;
  eoSelect<EOT>*    ptselect_mate;
  eoMerge<EOT>*     ptreplace;
  bool elitism;	     /* put back old best in the new population if necessary */
};
/* examples:
   for most populat schemes, nb_survive = nb_die = 0
   in GGA and EP, nb_offspring = pop.size()
   in ES, nb_offspring = lambda
   in SSGA, nb_offspring = 1 (usually)

   elitism can be used anywhere - though stupid in ES, EP and SSGA who are 
        elist by definition
*/

#endif _EOSCHEME_H

/* 
  /////////////////////////////////
  /// Applies one generation of evolution to the population.
  virtual void operator()(eoPop<EOT>& pop) {
    // Determine the number of offspring to create
    // either prescribed, or given as a rate
    unsigned nb_off_local = (nb_offspring ? nb_offspring : 
			 (int) rint (rate_offspring * pop.size()) );
    nb_off_local = (nb_off_local ? nb_off_local : 1);   // in case it is rounded to 0!
      
    // the worst die immediately
    unsigned nb_die_local = (nb_die ? nb_die : 
			 (int) rint (rate_die * pop.size()) );
    // and the best will survive without selection
    unsigned nb_survive_local = (nb_survive ? nb_survive : 
			 (int) rint (rate_survive * pop.size()) );

    // before selection, erase the one to die
    // sort old pop - just in case!
    sort(pop.begin(), pop.end(), greater<EOT>());
    Fitness oldBest = pop[0].fitness();    // store for elitism
    eoPop<EOT> fertilepop = pop;
    if (nb_die_local)
	erase(fertilepop.end()-nb_die_local, fertilepop.end());
    
    eoPop<EOT> offspring;    // = select(fertilepop, nb_off_local);
    select(fertilepop, offspring, nb_off_local);

    // now apply the operators to offspring
    for (unsigned i=0; i<nb_local; i++) {
      EOT tmp = offspring[i];
      unsigned id = 0;	   // first operator
      eoOp<EOT>* op = seqselop.Op(&id);
      while (op) {		   // NULL if no more operator
	EOT mate;
	if (op->readArity() == binary) // can eventually be skipped 
	  mate = select_mate(pop, tmp);	// though useless ig mutation
	else
	  mate = tmp;	   // assumed: mate will not be used!
	cout << op->className() << " for offspring " << i << endl;
	tmp = (*op)( tmp, mate, pop );  
	op = seqselop.Op(&id);
      }
      offspring[i]=tmp;		//* where it belongs
    }
    
    eoPop<EOT>::iterator i;
    // Can't use foreach here since foreach takes the 
    // parameter by reference
    for ( i = offspring.begin(); i != offspring.end(); i++)
      evaluator(*i);

    //XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    // not exact - later!!!
    // -MS-

      // first, copy the ones that need to survive
      // assumed: pop is still sorted!
      eoPop<EOT> finalPop;
    // and the best survive without selection
      if (nb_survive_local) {
	  finalPop.resize(nb_survive_local);
	  copy( finalPop.begin(), fertilepop.begin(), 
	        fertilepop.begin()+nb_survive_local );
      }

    // now call the replacement method
    replace(finalPop, tmpPop);

    // handle elitlism
    sort(finalPop.begin(), finalPop.end(), greater<EOT>());
    if (elitism) {
	if ( finalPop[0].fitness() < oldBest ) // best fitness has decreased!
	    copy(finalPop.end()-1, pop[0]);
    }
    //    return finalPop;
  }
*/
