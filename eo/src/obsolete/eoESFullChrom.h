// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoESInd.h
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
             Marc.Schoenauer@polytechnique.fr
 */
//-----------------------------------------------------------------------------

#ifndef _EOESFULLCHROM_H
#define _EOESFULLCHROM_H

// STL libraries
#include <vector>		// For std::vector<>
#include <stdexcept>
#include <strstream>
#include <iostream>		// for std::ostream

// EO includes
#include <eoVector.h>
#include <utils/eoRNG.h>
/**@name Chromosomes for evolution strategies
Each chromosome in an evolution strategies is composed of a std::vector of floating point
values plus a std::vector of sigmas, that are added to them during mutation and a std::vector of correlations
*/ 
//@{


/**@name individuals for evolution strategies  -MS- 22/10/99
Each individual in an evolution strategy is composed of 
   a std::vector of floating point values 
   a std::vector of std deviations
   a std::vector of rotation angles (for correlated mutations)

These individuals CANNOT BE IMPLEMENTED as std::vectors of anything 
      at least in the case of correlated mutations
*/ 
//@{

template <typename fitT = float >
class eoESFullChrom  : public eoVector<double, fitT> {	
    public:
/// constructor
    eoESFullChrom(  unsigned _num_genes = 1, 
	      unsigned _num_sigma = 1, unsigned _num_correl = 0,
	      bool _verbose = false,  
	      double _ObjMin = 0, double _ObjMax = 1, 
	      double _StdDevInit = 0.3 ):
    eoVector<double, fitT>(_num_genes),
    //    ObjVar( _num_genes ), now an eoVector<double>
    StdDev( _num_sigma ), 
    CorCff( _num_correl ),
    verbose( _verbose ),
    ObjMin( _ObjMin ),
    ObjMax(_ObjMax ),
    StdDevInit( _StdDevInit ) 
    { // check consistency
    }
  
	
  /* And now the useful constructor: from a parser (should be in the
     factory, if such a thing exists one day for eoESFullChrom 
   */
  eoESFullChrom(eoParser & parser) : StdDev(0), CorCff(0) {
    parser.AddTitle("Description of ES individuals");
    int num_genes, num_sigma;
    bool correlated_mutations;
    try {
      num_genes = parser.getInt("-Io", "--NbObjVar", "2", 
				 "Number of Object Variables" );
      num_sigma = parser.getInt("-Is", "--NbSigma", "1", 
				 "Number of Standard Deviations" );
      correlated_mutations = parser.getBool("-Ic", "--Correlated", 
				  "Correlated mutation?" );
      ObjMin = parser.getFloat("-Im", "--min", "0", 
			      "Minimum value for object variables" );
      ObjMax = parser.getFloat("-IM", "--max", "1", 
			      "Maximum value for object variables" );
      StdDevInit = parser.getFloat("-II", "--SigmaInit", "0.3", 
			       "Initial value for std. dev. (scaled by range)" );
      verbose = parser.getBool("-Iv", "--verbose",
		"Verbose std::listing of ES individuals (mutation parameters");
    }
    catch (std::exception & e)
      {
	std::cout << e.what() << std::endl;
	parser.printHelp();
	exit(1);
      }

    // consistency tests
    if (! num_sigma) {		   // no std dev??? EXCEPTION
      throw invalid_argument( "No standard deviation: choose another representation please" );
    }
    if (num_sigma > num_genes) {
      std::cout << "WARNING, Number of Standard Deviations > Number of Object Variables\nAdjusted!\n";
      num_sigma = num_genes;
      // modify the Param value - so .status is OK
      std::ostrstream sloc;
      sloc << num_genes;
      parser.setParamValue("--NbSigma", sloc.str());
    }
    // adjust the sizes!!!
    resize(num_genes);
    if (num_sigma)
      StdDev.resize(num_sigma);
    if (correlated_mutations) {
      if (num_sigma < num_genes) {
	std::cout << "WARNING less Std Dev. than number of variables + Correlated mutations\n";
	std::cout << "Though possible, this is a strange setting" << std::endl;
      }
      // nb of rotation angles: N*(N-1)/2 (in general!)
      CorCff.resize ( (2*num_genes - num_sigma)*(num_sigma - 1) / 2 );
    }
  };


  /// Operator =
  const eoESFullChrom& operator = ( const eoESFullChrom& _eo ) {
    if ( this  != &_eo ) {
      // Change EO part
      eoVector<double, fitT>::operator = (_eo);
      
      // Change this part
      //	      ObjVar = _eo.ObjVar;
      StdDev = _eo.StdDev;
      CorCff = _eo.CorCff;
      verbose =  _eo.verbose;
      ObjMin = _eo.ObjMin;
      ObjMax = _eo.ObjMax;
      StdDevInit = _eo.StdDevInit;
    }
    return *this;
  }

  /// destructor
  virtual ~eoESFullChrom() {}
  
  /// 
  double getStdDev( unsigned _i ) const {
    if ( _i >= length() ) 
      throw out_of_range( "out_of_range when reading StdDev");
    return StdDev[ _i ];
  }

  ///
  void setStdDev( unsigned _i, double _val ) {
    if ( _i < length() ) {
      StdDev[_i] = _val;
    } else 
      throw out_of_range( "out_of_range when writing StdDev");
  }
  
  /// 
  double getCorCff( unsigned _i ) const {
    if ( _i >= length() ) 
      throw out_of_range( "out_of_range when reading CorCff");
    return CorCff[ _i ];
  }

  ///
  void setCorCff( unsigned _i, double _val ) {
    if ( _i < length() ) {
      CorCff[_i] = _val;
    } else 
      throw out_of_range( "out_of_range when writing CorCff");
  }
  
  ///
  void insertGene( unsigned _i, double _val ) {
      throw FixedLengthChromosome();
  };
  
  ///
  void deleteGene( unsigned _i ) {
      throw FixedLengthChromosome();
  };
  
  ///
    unsigned length() const { return size();}/* formerly ObjVar.size() */
    unsigned StdDevLength() const { return StdDev.size();}
    unsigned CorCffLength() const { return CorCff.size();}


  /** Print itself: inherited from eoObject implementation. 
      Instance from base classes are processed in
      base classes, so you don´t have to worry about, for instance, fitness.
  @param _s the std::ostream in which things are written*/
  virtual void printOn( std::ostream& _s ) const{
      copy( begin(), end(), std::ostream_iterator<double>( _s, " ") );
      // The formatting instructinos shoudl be left to the caller
      //      _s << "\n";       
      if (verbose) {
	  _s << "\n\tStd Dev. " ;
	  copy( StdDev.begin(), StdDev.end(), std::ostream_iterator<double>( _s, " ") );
	  if (CorCff.size()) {
	      _s << "\n\t";
	      copy( CorCff.begin(), CorCff.end(), std::ostream_iterator<double>( _s, " ") );
	  }
      }
  };

  /** This std::exception should be thrown when trying to insert or delete a gene
  in a fixed length chromosome  
  */
  class FixedLengthChromosome : public std::exception {

  public:
    /**
       * Constructor
       */
    FixedLengthChromosome()
	: std::exception() { };

    ~FixedLengthChromosome() {};
  };
  
  // accessors
  double getObjMin() const {return ObjMin;}
  double getObjMax() const {return ObjMax;}
  double getStdDevInit () const {return StdDevInit;}

  /** Inherited from eoObject 
      @see eoObject
  */
  virtual std::string className() const {return "eoESFullChrom";};

private:
    //	std::vector<double>	    ObjVar;	/* object variable std::vector */
// or shoudl the class be subclass of EOVector<double> ???

	std::vector<double>	    StdDev;	/* standard deviation std::vector */
	std::vector<double>	    CorCff;	/* correlation coefficient std::vector */

    bool verbose;		/* Print std deviations or not */

  /** the range is used for mutation AND random initialization, 
     * while the StdDevInit is used only for random initialization
     * this in a little inconsistent!
     */
    double ObjMin, ObjMax;	/* Range for Object variables */
    double StdDevInit;		/* Initial value of Standard Deviations */

};




#endif


