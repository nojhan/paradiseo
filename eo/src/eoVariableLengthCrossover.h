// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoVariableLengthCrossover.h
// (c) GeNeura Team, 2000 - EEAAX 1999 - Maarten Keijzer 2000
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
             Marc.Schoenauer@polytechnique.fr
             mak@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef _eoVariableLengthCrossover_h
#define _eoVariableLengthCrossover_h

#include <eoFunctor.h>
#include <eoVariableLength.h>
#include <eoGenericBinOp.h>
#include <eoGenericQuadOp.h>

/**
  Base classes for generic crossovers on variable length chromosomes. 

Basically, they exchange genes - we need some matching information to apply 
atom crossover
*/

/** A helper class for choosing which genes to exchange */
template <class Atom>
class eoAtomExchange : public eoBF<unsigned, Atom &, bool>
{
public:
  // a function to initlialize - to be called before every crossover
  virtual void randomize(unsigned int, unsigned int){}
};

/** Uniform crossover - well, not really for FixedLength */
template <class Atom>
class eoUniformAtomExchange: public eoAtomExchange<Atom>
{
public:
    eoUniformAtomExchange(double _rate=0.5):rate(_rate){}

  // randomize: fill the mask: the exchange will be simulated first
  // to see if sizes are OK, so it must be repeatable
  void randomize(unsigned _size1, unsigned _size2)
  {
    mask.resize(_size1 + _size2);
    for (unsigned i=0; i<_size1+_size2; i++)
	mask[i]=eo::rng.flip(rate);
  }

  // the operator() simply returns the mask booleans in turn
    bool operator()(unsigned _i, Atom & )
    {
      return mask[_i];
    }
private:
  double rate;
  vector<bool> mask;
};


/** Exchange Crossover using an AtomExchange
 */

template <class EOT>
class eoVlAtomExchangeQuadOp : public eoGenericQuadOp<EOT>
{
public :
	
  typedef typename EOT::AtomType AtomType;
  
  // default ctor: requires bounds on number of genes + a rate
  eoVlAtomExchangeQuadOp(unsigned _Min, unsigned _Max, 
			 eoAtomExchange<AtomType>& _atomExchange): 
    Min(_Min), Max(_Max), atomExchange(_atomExchange) {}
  
  bool operator()(EOT & _eo1, EOT & _eo2)
  {
    EOT tmp1, tmp2;		   // empty individuals
    unsigned index=0;
    // main loop: until sizes are OK, do only simulated exchange
    unsigned i, i1, i2;
    do {
      // "initialize the AtomExchange
      atomExchange.randomize(_eo1.size(), _eo2.size());
      // simulate crossover
      i1=i2=0;
      for (i=0; i<_eo1.size(); i++)
	{
	  if (atomExchange(i, _eo1[i]))
	    i1++;
	  else
	    i2++;
	}
      for (i=0; i<_eo2.size(); i++)
	{
	  if (atomExchange(i, _eo2[i]))
	    i2++;
	  else
	    i1++;
	}
      index++;
    } while ( ( (i1<Min) || (i2<Min) || 
		(i1>Max) || (i2>Max) ) 
	      && (index<10000) );
    if (index >= 10000)
      {
	cout << "Warning: impossible to generate individual of the right size in 10000 trials\n";
	return false;
      }
  // here we know we have the right sizes: do the actual exchange
      for (i=0; i<_eo1.size(); i++)
	{
	  if (atomExchange(i, _eo1[i]))
	    tmp1.push_back(_eo1[i]);
	  else
	    tmp2.push_back(_eo1[i]);
	}
      for (i=0; i<_eo2.size(); i++)
	{
	  if (atomExchange(i, _eo2[i]))
	    tmp2.push_back(_eo2[i]);
	  else
	    tmp1.push_back(_eo2[i]);
	}
      // and put everything back in place
    _eo1.swap(tmp1);
    _eo2.swap(tmp2);
    return true;	 // should we test that? Yes, but no time now
  }
private:
  unsigned Min, Max;
  eoAtomExchange<AtomType> & atomExchange;
};




/** Direct Uniform Exchange of genes (obsolete, already :-)

A very primitive version, that does no verification at all!!!
NEEDS to be improved - but no time now :-(((
Especially, if both guys have maximal size, it will take a lot of time
to generate 2 offspring that both are not oversized!!! 
Also, we should first check for identical atoms, and copy them to the
offspring, and only after that exchange the other ones (Radcliffe's RRR).
 */
template <class EOT>
class eoVlUniformQuadOp : public eoGenericQuadOp<EOT>
{
public :
	
  typedef typename EOT::AtomType AtomType;
  
  // default ctor: requires bounds on number of genes + a rate
  eoVlUniformQuadOp(unsigned _Min, unsigned _Max, double _rate=0.5) : 
    Min(_Min), Max(_Max), rate(_rate) {}
  
  bool operator()(EOT & _eo1, EOT & _eo2)
  {
    unsigned i;
    EOT tmp1, tmp2;
    unsigned index=0;
    do {
      for (i=0; i<_eo1.size(); i++)
	{
	  if (eo::rng.flip(rate))
	    tmp1.push_back(_eo1[i]);
	  else
	    tmp2.push_back(_eo1[i]);
	  // here we should look for _eo1[i] inside _eo2 and erase it if found!
	}
      for (i=0; i<_eo2.size(); i++)
	{
	  if (eo::rng.flip(rate))
	    tmp1.push_back(_eo2[i]);
	  else
	    tmp2.push_back(_eo2[i]);
	}
      index++;
    } while ( ( (tmp1.size()<Min) || (tmp2.size()<Min) || 
	      (tmp1.size()>Max) || (tmp2.size()>Max) )
	      && (index<10000) );
    if (index >= 10000)
      {
	cout << "Warning: impossible to generate individual of the right size in 10000 trials\n";
	return false;
      }

    _eo1.swap(tmp1);
    _eo2.swap(tmp2);
    return true;		   // should we test that?
  }
private:
  unsigned Min, Max;
  double rate;
};


/** Direct Uniform Exchange of genes for Variable Length, BINARY version

A very primitive version, that does no verification at all!!!
NEEDS to be improved - but no time now :-(((
Especially, if both guys have maximal size, it will take some time
to generate even 1 offspring that is not oversized!!! 
Also, we should first check for identical atoms, and copy them to the
offspring, and only after that exchange the other ones (Radcliffe's RRR).
 */
template <class EOT>
class eoVlUniformBinOp : public eoGenericBinOp<EOT>
{
public :

  typedef typename EOT::AtomType AtomType;
  
  // default ctor: requires bounds on number of genes + a rate
  eoVlUniformBinOp(unsigned _Min, unsigned _Max, double _rate=0.5) : 
    Min(_Min), Max(_Max), rate(_rate) {}
  
  bool operator()(EOT & _eo1, const EOT & _eo2)
  {
    unsigned i;
    EOT tmp1;
    bool tmpIsOne=true, tmpIsTwo=true;
    unsigned index=0;
    do {
      for (i=0; i<_eo1.size(); i++)
	{
	  if (eo::rng.flip(rate))
	    {
	      tmp1.push_back(_eo1[i]);
	      tmpIsTwo = false;
	    }
	  else
	    tmpIsOne=false;	    
	// we should look for _eo1[i] inside _eo2 and erase it there if found!
	}
      for (i=0; i<_eo2.size(); i++)
	{
	  if (! eo::rng.flip(rate))
	    {
	      tmp1.push_back(_eo2[i]);
	      tmpIsOne = false;
	    }
	  else
	    tmpIsTwo = false;
	}
      index++;
    } while ( ( (tmp1.size()<Min) || (tmp1.size()>Max) )
	      && (index<10000) );
    // this while condition is not optimal, as it may take some time
    if (index >= 10000)
      {
	cout << "Warning: impossible to generate individual of the right size in 10000 trials\n";
	return false;
      }

    _eo1.swap(tmp1);
    if (tmpIsTwo)
      {
	//	_eo1.fitness(_eo2.fitness());     NO FITNESS EXISTS HERE!
	return false;
      }
    if (tmpIsOne)		   // already has the right fitness
      {				   // WRONG: NO FITNESS EXISTS HERE!
	return false;
      }
    return true;		   // there were some modifications...
  }

private:
  unsigned Min, Max;
  double rate;
};



#endif
