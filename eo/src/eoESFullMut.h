// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-
 
//-----------------------------------------------------------------------------
// eoESMute.h : ES mutation
// (c) GeNeura Team, 1998 for the EO part
//     Th. Baeck 1994 and EEAAX 1999 for the ES part
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
             marc.schoenauer@polytechnique.fr 
                       http://eeaax.cmap.polytchnique.fr/
 */
//-----------------------------------------------------------------------------


#ifndef _EOESMUT_H
#define _EOESMUT_H

#include <eoParser.h>
#include <eoRNG.h>
#include <cmath>		// for exp

#include <eoESFullChrom.h>
#include <eoOp.h>

const double ES_SIGEPS = 1.0e-40;       	/* ES lower bound for sigma values */
// should not be a parameter ...

/** ES-style mutation in the large: Obviously, valid only for eoESInd
*/
template <class fitT>
class eoESMutate: public eoMonOp< eoESFullChrom<fitT> > {
public:
    ///
    eoESMutate( double _TauLcl, double _TauGlb, double _TauBeta )
	: eoMonOp< eoESFullChrom<fitT> >( ), TauLcl(_TauLcl), TauGlb(_TauGlb),
	  TauBeta(_TauBeta) {};

    /* The parser constructor
     */
    eoESMutate(Parser & parser, unsigned _stdDevLength, unsigned _size, bool _correlated ): 
	eoMonOp< eoESFullChrom<fitT> >( ) {
      parser.AddTitle("Parameters of ES mutation (before renormalization)");
      try {			// we know that there is at least 1 std dev.
	if (_stdDevLength == 1) {
	  TauLcl = parser.getInt("-Ml", "--TauLcl", "1", 
				 "TauLcl, Mutation rate for the only Std Dev." );
	  // different normalization in that case -- Thomas Baeck
	  TauLcl /= sqrt((double) _size);
	} 
	else {		/* more than 1 std dev */
	  TauLcl = parser.getFloat("-Ml", "--TauLcl", "1", 
				 "Local mutation rate for Std Dev." );
	  TauGlb = parser.getFloat("-Mg", "--TauGlb", "1", 
				"Global mutation rate for Std Dev." );
	  // renormalization
	  TauLcl /= sqrt( 2.0 * sqrt( (double)_size ) );
	  TauGlb /= sqrt( 2.0 * ( (double) _size ) );

	  if ( _correlated ) { // Correlated Mutations
	      TauBeta = parser.getFloat("-Mb", "--TauBeta", "0.0873", 
				 "Mutation rate for corr. coeff." );
	      // rotation angles: no normalization 
	  }
	}
      }
      catch (exception & e)
	{
	  cout << e.what() << endl;
	  parser.printHelp();
	  exit(1);
	}
    };

    /// needed virtual dtor
    virtual ~eoESMutate() {};
    
    // virtual separation depending wether correlated mutations are present
    virtual void operator() (  eoESFullChrom<fitT> & _eo ) const {
      if (_eo.CorCffLength())
	CorrelatedMutation(_eo);
      else
	StandardMutation(_eo);
    }

  /** @name Methods from eoObject
      readFrom and printOn are directly inherited from eoObject
  */
  //@{
  /** Inherited from eoObject 
      @see eoObject
  */
  virtual string className() const {return "eoESMutate";};
  
private:
  /// mutations - standard et correlated
  //  ========= 
  /*
   *	Standard mutation of object variables and standard 	
   *	deviations in ESs. 
   *	If there are fewer different standard deviations available 
   *	than the dimension of the objective function requires, the 
   * 	last standard deviation is responsible for ALL remaining
   *	object variables.
   *	Schwefel 1977: Numerische Optimierung von Computer-Modellen
   *	mittels der Evolutionsstrategie, pp. 165 ff.
   */
  
  virtual void StandardMutation( eoESFullChrom<fitT> & _eo ) const {
    unsigned i,k;
    double Glb, StdLoc;
    
    if (_eo.StdDevLength() == 1) { /* single StdDev -> No global factor */
      StdLoc = _eo.getStdDev(0);
      StdLoc *= exp(TauLcl*rng.normal());	
      if (StdLoc < ES_SIGEPS)
	StdLoc = ES_SIGEPS;
      _eo.setStdDev(0, StdLoc);
      _eo.setGene( 0, _eo.getGene(0) + StdLoc*rng.normal());
      i = 1;
    }
    else {			/* more than one std dev. */
      Glb = exp(TauGlb*rng.normal());
      for (i = 0; i < _eo.length() && i < _eo.StdDevLength(); i++) {
	StdLoc = _eo.getStdDev(i);
	StdLoc *= Glb * exp(TauLcl*rng.normal());	
	if (StdLoc < ES_SIGEPS)
	  StdLoc = ES_SIGEPS;
	_eo.setStdDev(i, StdLoc);
	_eo.setGene( i, _eo.getGene(i) + StdLoc*rng.normal());
      }
    }
    // last object variables: same STdDev than the preceding one
    for (k = i; k < _eo.length(); k++) {
      _eo.setGene( k, _eo.getGene(k) + StdLoc*rng.normal() );
    }
  }

  /*
   *	Correlated mutations in ESs, according to the following
   *	sources:
   *	H.-P. Schwefel: Internal Report of KFA Juelich, KFA-STE-IB-3/80
   *	p. 43, 1980
   *	G. Rudolph: Globale Optimierung mit parallelen Evolutions-
   *	strategien, Diploma Thesis, University of Dortmund, 1990
   */
  
  // Code from Thomas Baeck 
  
  virtual void CorrelatedMutation( eoESFullChrom<fitT> & _eo ) const {

    
    int		i, k, n1, n2, nq;
    
    double		d1, d2, S, C, Glb;
    double tmp;
    /*
     *	First: mutate standard deviations (as above).
     */
    
    Glb = exp(TauGlb*rng.normal());
    for (i = 0; i < _eo.StdDevLength(); i++) {
      tmp = _eo.getStdDev(i);
      _eo.setStdDev( i, tmp*Glb*exp(TauLcl*rng.normal()) );
    }
    
    /*
     *	Mutate rotation angles.
     */
    
    for (i = 0; i < _eo.CorCffLength(); i++) {
      tmp = _eo.getCorCff(i);
      tmp += TauBeta*rng.normal();
      // danger of VERY long loops --MS--
      // 		while (CorCff[i] > M_PI)
      // 			CorCff[i] -= 2.0 * M_PI;
      // 		while (CorCff[i] < - M_PI)
      // 			CorCff[i] += 2.0 * M_PI;
      if ( fabs(tmp) > M_PI ) {
	tmp -= M_PI * (int) (tmp/M_PI) ;
      }
      _eo.setCorCff(i, tmp);
    }
    
    /*
     *	Perform correlated mutations.
     */
    vector<double> VarStp(_eo.size());
    for (i = 0; i < _eo.size() && i < _eo.StdDevLength(); i++) 
      VarStp[i] = _eo.getStdDev(i)*rng.normal();
    for (k = i; k < _eo.size(); k++) 
      VarStp[k] = _eo.getStdDev(i-1)*rng.normal();
    nq = _eo.CorCffLength() - 1;
    for (k = _eo.size()-_eo.StdDevLength(); k < _eo.size()-1; k++) {
      n1 = _eo.size() - k - 1;
      n2 = _eo.size() - 1;
      for (i = 0; i < k; i++) {
	d1 = VarStp[n1];
	d2 = VarStp[n2];
	S  = sin( _eo.getCorCff(nq) );
	C  = cos( _eo.getCorCff(nq) );
	VarStp[n2] = d1 * S + d2 * C;
	VarStp[n1] = d1 * C - d2 * S;
	n2--;
	nq--;
      }
    }
    for (i = 0; i < _eo.size(); i++) 
      _eo[i] += VarStp[i];
    
  }
  // the data
  //=========
  double TauLcl;	/* Local factor for mutation of std deviations */
  double TauGlb;	/* Global factor for mutation of std deviations */
  double TauBeta;	/* Factor for mutation of correlation parameters  */
};

/*
 *	Correlated mutations in ESs, according to the following
 *	sources:
 *	H.-P. Schwefel: Internal Report of KFA Juelich, KFA-STE-IB-3/80
 *	p. 43, 1980
 *	G. Rudolph: Globale Optimierung mit parallelen Evolutions-
 *	strategien, Diploma Thesis, University of Dortmund, 1990
 */
// Not yet implemented!

#endif

