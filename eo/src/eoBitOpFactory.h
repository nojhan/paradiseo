// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoOpFactory.h
// (c) GeNeura Team, 1998
//-----------------------------------------------------------------------------

#ifndef _EOBITOPFACTORY_H
#define _EOBITOPFACTORY_H

#include <eoOpFactory.h>
#include <eoBitOp.h>

//-----------------------------------------------------------------------------

/** EO Factory. An instance of the factory class to create operators that act
on bitstring chromosomes. Only those chromosomes can instantiate the operators
that are created here
@see eoSelect*/
template< class EOT>
class eoBitOpFactory: public eoOpFactory<EOT>  {
	
public:
	
	/// @name ctors and dtors
	//{@
	/// constructor
	eoBitOpFactory( ) {};
	
	/// destructor
	virtual ~eoBitOpFactory() {};
	//@}

	/** Another factory method: creates an object from an istream, reading from
	it whatever is needed to create the object. Usually, the format for the istream will be\\
	objectType parameter1 parameter2 ... parametern\\
	If there are problems, an exception is raised; it should be caught at the
	upper level, because it might be something for that level\\
	At the same time, it catches exceptions thrown at a lower level, which will
	indicate that whatever is in the stream is for this method to process
	@param _is an stream from where a single line will be read
	@throw runtime_exception if the object type is not known
	*/
	virtual eoOp<EOT>* make(istream& _is) {
		eoOp<EOT> * opPtr = NULL;
		try {
			opPtr = eoOpFactory<EOT>::make( _is );
		} catch ( string& objectTypeStr ) {
			if  ( objectTypeStr == "eoBinRandom") {
				opPtr = new eoBinRandom<EOT>();
			} 
			if ( objectTypeStr == "eoBinBitFlip" ) {
				opPtr = new eoBinBitFlip<EOT>( );
			} 
			if ( objectTypeStr == "eoBinMutation" ) {
				float rate;
				_is >> rate;
				opPtr = new eoBinMutation<EOT>( rate );
			} 
			if ( objectTypeStr == "eoBinInversion" ) {
				opPtr = new eoBinInversion<EOT>( );
			}
			if ( objectTypeStr == "eoBinNext" ) {
				opPtr = new eoBinNext<EOT>( );
			}
			if ( objectTypeStr == "eoBinPrev" ) {
				opPtr = new eoBinPrev<EOT>( );
			}
			if ( objectTypeStr == "eoBinNext" ) {
				opPtr = new eoBinNext<EOT>( );
			}
			if ( objectTypeStr == "eoBinCrossover" ) {
				opPtr = new eoBinCrossover<EOT>( );
			}
			if ( objectTypeStr == "eoBinNxOver" ) {
				unsigned nPoints;
				_is >> nPoints;
				opPtr = new eoBinNxOver<EOT>( nPoints );
			}
			if ( objectTypeStr == "eoBinGxOver" ) {
				unsigned geneSize, nPoints;
				_is >> geneSize >> nPoints;
				opPtr = new eoBinGxOver<EOT>( geneSize, nPoints );
			}
			if ( objectTypeStr == "eoBinUxOver" ) {
				float rate;
				_is >> rate;
				opPtr = new eoBinUxOver<EOT>( rate );
			}
			if ( !opPtr ) {	// to be caught by the upper level
				throw objectTypeStr;
			}
		}
		return opPtr;
	};


};


#endif _EOBITOPFACTORY_H
