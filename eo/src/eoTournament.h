// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoTournament.h
// (c) GeNeura Team, 1998
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

#ifndef _EOGSTOURN_H
#define _EOGSTOURN_H

//-----------------------------------------------------------------------------

#include <eoUniform.h>                       // for ceil
#include <eoPopOps.h>

//-----------------------------------------------------------------------------

/** Selects those who are going to reproduce using Tournament selection: 
	a subset of the population of size tournamentSize is chosen, 
	and the best is selected for the new population .
@author JJ Merelo, 1998
*/
template<class EOT>
class eoTournament:public eoBinPopOp<EOT>{
public:

  /// Proportion of guys that are going to be eliminated
  eoTournament( float _perc, unsigned _tSize): 
    eoBinPopOp<EOT>(), perc( _perc), repTournamentSize(_tSize){};

  /// Virtual dtor
  ~eoTournament(){};

  /// Set tourn size
  void tournamentSize( unsigned _size ) { repTournamentSize = _size; };

  /** 
   * Selects from the initial pop using tournament selection, and copies it
   * to the other population.
   */
  virtual void operator() ( eoPop<EOT>& _vEO, eoPop<EOT>& _aVEO)  {
    
    unsigned thisSize = _vEO.size();
    
    // Build vector
    for ( unsigned j = 0; j < thisSize*perc; j ++ ) {
      // Randomly select a tournamentSize set, and choose the best
      eoPop<EOT> veoTournament;
      eoUniform<unsigned> u( 0, thisSize);
      for ( unsigned k = 0; k < repTournamentSize; k++ ) {
	unsigned chosen = u();
	EOT newEO =  _vEO[chosen];
	veoTournament.push_back( newEO );
      }
      
      eoPop<EOT>::const_iterator best = max_element(veoTournament.begin(),
						    veoTournament.end());
      if (best == veoTournament.end()) {
	throw runtime_error("error in void eoTournament::operator(eoPop<EOT>&, eoPop<EOT>&)");
      }
      
      // The first is chosen for the new population
      _aVEO.push_back( best ); 
    }
  };
  
  /// @name Methods from eoObject
  //@{
  /**
   * Read object. Reads the percentage
   * Should call base class, just in case.
   * @param _s A istream.
   */
  virtual void readFrom(istream& _s) {
	_s >> perc >> repTournamentSize;
  }

  /** Print itself: inherited from eoObject implementation. Declared virtual so that 
      it can be reimplemented anywhere. Instance from base classes are processed in
	  base classes, so you don´t have to worry about, for instance, fitness.
  @param _s the ostream in which things are written*/
  virtual void printOn( ostream& _s ) const{
	_s << perc << endl << repTournamentSize << endl;
  }

  /** Inherited from eoObject 
      @see eoObject
  */
  string className() const {return "eoTournament";};

  //@}

 private:
	 float perc;
  unsigned repTournamentSize;
  
};

#endif
