// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoRandomBreed.h
// (c) GeNeura Team, 1998
//-----------------------------------------------------------------------------

#ifndef _EORANDOMBREED_H
#define _EORANDOMBREED_H

#include <eoSelector.h>

/** Takes those on the selection list and creates a list of new individuals
 * Destroys the genetic pool */
template<class EOT>
class EORandomBreed: public EOBreeder<EOT>{
public:

  typedef vector< EOOp<EOT > * > vecOpT;

  /// Ctor
  EORandomBreed():vecOp() {};

  /** Copy ctor
   * Needs a copy ctor for the EO operators */
  EORandomBreed( const EORandomBreed&  _rndBreeder)
    :vecOp() {
      copy( _rndBreeder.vecOp.begin(), _rndBreeder.vecOp.end(), 
	    vecOp.begin() );
  };

  /// Dtor
  virtual ~EORandomBreed() {};

  /// Adds a genetic operator to the Breeder with a rate
  virtual void addOp( EOOp<EOT>* _eop )  {	
    vecOp.push_back( _eop);
  };

  /// Takes the operator pointed to from the operator list
  virtual void deleteOp( const EOOp<EOT>* _eop);

  /** Takes the genetic pool, and returns next generation, destroying the
   * genetic pool container
   * Non-const because it might order the operator vector. In this case, 
   * it mates all members of the population randomly */
  virtual void operator() ( EOPop< EOT >& _ptVeo );

// I don´t like this, but I need it for the RandomBreedLog
protected:
  vecOpT vecOp;

};


//_________________________ IMPLEMENTATIONS _____________________________

template< class EOT>
void EORandomBreed<EOT>::deleteOp( const EOOp<EOT>* _eop)  {
    vecOpT::iterator i;
    for ( i = vecOp.begin(); i != vecOp.end(); i++ ) {
		if ( *i == _eop ) {
			vecOp.erase( i );
		}
    }
}

//________________________________________________________________________
template<class EOT>
void EORandomBreed<EOT>::operator() ( EOPop< EOT >& _ptVeo )  { 
	
    unsigned select= _ptVeo.size(); // New population same size than old
    sort( vecOp.begin(), vecOp.end(), SortEOpPt<EOT>() );
	
    unsigned i;
    float totalPriority = 0;
    for ( i = 0; i < vecOp.size(); i ++ ) {
      totalPriority += vecOp[i]->Priority();
    }
    
    unsigned inLen = _ptVeo.size(); // size of in subPop
    for ( i = 0; i < select; i ++ ) {
      // Create an alias of a random input EO with copy ctor
      EOT& newEO = _ptVeo[ i ];
      
      // Choose operator
      float randomDraw = totalPriority *(rand() % 1000) /1000.0;
      vecOpT::const_iterator j;
      float accumulated = 0;
      for ( j = vecOp.begin(); 
	    ( j != vecOp.end() ) && ( accumulated < randomDraw); 
	    j ++ ) {
	accumulated+= (*j)->Priority(); // the priority
      }
      
      if ( j != vecOp.begin() ) 
	j--;			// previous one
      EOOp<EOT >* thisOp = *j;
      
      if (thisOp->readArity() == unary ) {
	MonOp<EOT>* mopPt = dynamic_cast< MonOp<EOT>* > ( thisOp );
	(*mopPt)( newEO );
      } else {
	unsigned chosenMatch = rand() % inLen;
	BinOp<EOT>* bopPt = dynamic_cast< BinOp<EOT>* > ( thisOp );
	(*bopPt)( newEO, _ptVeo[chosenMatch] );
      }
      
    }	
}

#include <ADT/EOFactory.h>		// For factory

/** Exactly as RandomBreed, except that uses  factories*/
template<class EOT>
class EORandomBreedLog: public EORandomBreed<EOT>{
public:

  typedef vector< EOOp<EOT > * > vecOpT;

  /// Ctor
  EORandomBreedLog( EOFactory<EOT> & _eof ):EORandomBreed<EOT>(), factory( _eof ) {};

  /** Copy ctor
   * Needs a copy ctor for the EO operators */
  EORandomBreedLog( const EORandomBreedLog&  _rndBreeder)
	  :EORandomBreed<EOT>( _rndBreeder ), factory( _rndBreeder.factory) {};

  /// Dtor
  virtual ~EORandomBreedLog() {};

  /** Takes the genetic pool, and returns next generation, destroying the
  * genetic pool container
  * Non-const because it might order the operator vector. In this case, it mates
  * all population randomly */
  virtual void operator() ( EOPop< EOT >& _ptVeo )  { 
	  
	  unsigned select= _ptVeo.size(); // New population same size than old	  
	  sort( vecOp.begin(), vecOp.end(), SortEOpPt<EOT>() );
	  
	  unsigned i;
	  float totalPriority = 0;
	  for ( i = 0; i < vecOp.size(); i ++ ) {
		  totalPriority += vecOp[i]->Priority();
	  }
	  
	  unsigned inLen = _ptVeo.size(); // size of in subPop
	  for ( i = 0; i < select; i ++ ) {
		  // Create a copy of a random input EO with copy ctor
		  EOT* newEO = factory.make( _ptVeo[ i ] );
		  
		  // Choose operator
		  float randomDraw = totalPriority *(rand() % 1000) /1000.0;
		  vecOpT::const_iterator j;
		  float accumulated = 0;
		  for ( j = vecOp.begin(); 
		  ( j != vecOp.end() ) && ( accumulated < randomDraw); 
		  j ++ ) {
			  accumulated+= (*j)->Priority(); // the priority
		  }
		  
		  if ( j != vecOp.begin() ) 
			  j--;			// previous one
		  EOOp<EOT >* thisOp = *j;
		  
		  if (thisOp->readArity() == unary ) {
			  MonOp<EOT>* mopPt = dynamic_cast< MonOp<EOT>* > ( thisOp );
			  (*mopPt)( *newEO );
		  } else {
			  unsigned chosenMatch = rand() % inLen;
			  BinOp<EOT>* bopPt = dynamic_cast< BinOp<EOT>* > ( thisOp );
			  (*bopPt)( *newEO, _ptVeo[chosenMatch] );
		  }
		  
	  }

  };
private:
  EOFactory<EOT>& factory;
 
};

#endif
