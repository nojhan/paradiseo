// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoRank.h
// (c) GeNeura Team 1999
//-----------------------------------------------------------------------------

#ifndef _eoRank_H
#define _eoRank_H

#include <eoOpSelector.h>
#include <eoPopOps.h>

/**
 * Takes those on the selection list and creates a list of new individuals
 * Destroys the genetic pool. There's no requisite on EOT, other than the 
 * genetic operators can be instantiated with it, so it fully depends on 
 * the genetic operators used. If generic genetic operators are used, then 
 * EOT must be an EO 
 */

template<class EOT>
class eoRank: public eoSelect<EOT>{
 public:
  
  /// Ctor
  eoRank( unsigned _newPopSize, eoOpSelector<EOT>& _opSelector)
	  :eoSelect<EOT>(), opSelector( _opSelector ), repNewPopSize( _newPopSize ) {};
  
  /** Copy ctor
   * Needs a copy ctor for the EO operators */
  eoRank( const eoRank&  _rankBreeder)
    :eoSelect<EOT>( _rankBreeder), 
	opSelector( _rankBreeder.opSelector ), repNewPopSize( _rankBreeder.repNewPopSize ) {};
  
  /// Dtor
  virtual ~eoRank() {};
  
  /** Takes the genetic pool, and returns next generation, destroying the
   * genetic pool container
   * Non-const because it might order the operator vector*/
  virtual void operator() (	const eoPop< EOT >& _ptVeo, 
							eoPop< EOT >& _siblings  ) const { 
    
    unsigned inLen = _ptVeo.size(); // size of subPop
    if ( !inLen ) 
      throw runtime_error( "zero population in eoRank");

    for ( unsigned i = 0; i < repNewPopSize; i ++ ) {
      // Create a copy of a random input EO with copy ctor. The members of the
		// population will be selected by rank, with a certain probability of
		// being selected several times if the new population is bigger than the
		// old
		EOT newEO =  _ptVeo[ i%inLen ];
    
		// Choose operator
		eoUniform<unsigned> u( 0, inLen );
		const eoOp<EOT >& thisOp = opSelector.Op();
		if ( thisOp.readArity() == unary ) {
			const eoMonOp<EOT>& mopPt = dynamic_cast< const eoMonOp<EOT>& > ( thisOp );
			mopPt( newEO );
		} else {
			const eoBinOp<EOT>& bopPt = dynamic_cast< const eoBinOp<EOT>& > ( thisOp );
			EOT mate =  _ptVeo[ u() ];
			bopPt( newEO, mate );
		}      
		
		_siblings.push_back( newEO );
    }
  };

  /** This is a _new_ function, non defined in the parent class
   * It´s used to set the selection rate */
  void select( unsigned _select ) {
    repNewPopSize = _select;
  }

  
      /// Methods inherited from eoObject
    //@{

    /** Return the class id. 
      @return the class name as a string
      */
    virtual string className() const { return "eoRank"; };

    /** Print itself: inherited from eoObject implementation. Declared virtual so that 
      it can be reimplemented anywhere. Instance from base classes are processed in
      base classes, so you don´t have to worry about, for instance, fitness.
      @param _s the ostream in which things are written*/
    virtual void printOn( ostream& _s ) const{
      _s << opSelector;
		_s << repNewPopSize;
    };


    //@}

private:
  eoOpSelector<EOT> & opSelector;
  unsigned repNewPopSize;
  
};

#endif
