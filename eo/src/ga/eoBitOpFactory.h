// eoBitOpFactory.h
// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoOpFactory.h
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

#ifndef _EOBITOPFACTORY_H
#define _EOBITOPFACTORY_H

#include <eoFactory.h>
#include <ga/eoBitOp.h>

//-----------------------------------------------------------------------------

/** EO Factory. An instance of the factory class to create operators that act
on bitstring chromosomes. Only those chromosomes can instantiate the operators
that are created here
@see eoSelect

@ingroup Variators
*/
template< class EOT>
class eoBitOpFactory: public eoFactory<EOT>
{

public:

        /// @name ctors and dtors
        //{@
        /// constructor
        eoBitOpFactory( ) {};

        /// destructor
        virtual ~eoBitOpFactory() {};
        //@}

        /** Another factory method: creates an object from an std::istream, reading from
        it whatever is needed to create the object. Usually, the format for the std::istream will be\\
        objectType parameter1 parameter2 ... parametern\\
        If there are problems, an std::exception is raised; it should be caught at the
        upper level, because it might be something for that level\\
        At the same time, it catches std::exceptions thrown at a lower level, which will
        indicate that whatever is in the stream is for this method to process
        @param _is an stream from where a single line will be read
        @throw runtime_std::exception if the object type is not known
        */
        virtual eoOp<EOT>* make(std::istream& _is)
    {
                eoOp<EOT> * opPtr = NULL;
                try {
                        opPtr = eoFactory<EOT>::make( _is );
                } catch ( const std::string& objectTypeStr ) {
                        if ( objectTypeStr == "eoBinBitFlip" ) {
                                opPtr = new eoOneBitFlip<EOT>( );
                        }
                        // handles old operator names as well as new ones
                        if ( objectTypeStr == "eoOneBitFlip" ) {
                                opPtr = new eoOneBitFlip<EOT>( );
                        }

                        // Standard BitFilp Mutation
                        if ( objectTypeStr == "eoBinMutation" ) {
                                float rate;
                                _is >> rate;
                                opPtr = new eoBitMutation<EOT>( rate );
                        }
                        if ( objectTypeStr == "eoBitMutation" ) {
                                float rate;
                                _is >> rate;
                                opPtr = new eoBitMutation<EOT>( rate );
                        }

                        // Bit inversion
                        if ( objectTypeStr == "eoBinInversion" ) {
                                opPtr = new eoBitInversion<EOT>( );
                        }
                        if ( objectTypeStr == "eoBitInversion" ) {
                                opPtr = new eoBitInversion<EOT>( );
                        }

                        // Next binary value
                        if ( objectTypeStr == "eoBinNext" ) {
                                opPtr = new eoBitNext<EOT>( );
                        }
                        if ( objectTypeStr == "eoBitNext" ) {
                                opPtr = new eoBitNext<EOT>( );
                        }

                        // Previous binary value
                        if ( objectTypeStr == "eoBinPrev" ) {
                                opPtr = new eoBitPrev<EOT>( );
                        }
                        if ( objectTypeStr == "eoBitPrev" ) {
                                opPtr = new eoBitPrev<EOT>( );
                        }

                        // 1 point Xover
                        if ( objectTypeStr == "eoBinCrossover" ) {
                                opPtr = new eo1PtBitXover<EOT>( );
                        }
                        if ( objectTypeStr == "eo1PtBitXover" ) {
                                opPtr = new eo1PtBitXover<EOT>( );
                        }

                        // Npts Xover
                        if ( objectTypeStr == "eoBinNxOver" ) {
                                unsigned nPoints;
                                _is >> nPoints;
                                opPtr = new eoNPtsBitXover<EOT>( nPoints );
                        }
                        if ( objectTypeStr == "eoNPtsBitXover" ) {
                                unsigned nPoints;
                                _is >> nPoints;
                                opPtr = new eoNPtsBitXover<EOT>( nPoints );
                        }

                        // Gene Xover (obsolete)
                        if ( objectTypeStr == "eoBinGxOver" ) {
                                unsigned geneSize, nPoints;
                                _is >> geneSize >> nPoints;
                                opPtr = new eoBitGxOver<EOT>( geneSize, nPoints );
                        }
                        if ( objectTypeStr == "eoBitGxOver" ) {
                                unsigned geneSize, nPoints;
                                _is >> geneSize >> nPoints;
                                opPtr = new eoBitGxOver<EOT>( geneSize, nPoints );
                        }

                        // Uniform Xover
                        if ( objectTypeStr == "eoBinUxOver" ) {
                                float rate;
                                _is >> rate;
                                opPtr = new eoUBitXover<EOT>( rate );
                        }
                        if ( objectTypeStr == "eoUBitXover" ) {
                                float rate;
                                _is >> rate;
                                opPtr = new eoUBitXover<EOT>( rate );
                        }

                        // nothing read!
                        if ( !opPtr ) {	// to be caught by the upper level
                                throw objectTypeStr;
                        }
                }
                return opPtr;
        };


};


#endif
