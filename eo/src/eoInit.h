// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoInit.h
// (c) Maarten Keijzer 2000, GeNeura Team, 2000
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

#ifndef _eoInit_H
#define _eoInit_H

#include <eoOp.h>

/**
	Base (name) class for Initialization of chromosomes, used in a population 
	contructor. It is derived from eoMonOp, so it can be used
    inside the algorithm as well.

	@see eoPop
*/

template <class EOT>
class eoInit : public eoUF<EOT&, void>
{};

/**
    Initializor for fixed length representations with a single type
*/
template <class EOT, class Gen>
class eoInitFixedLength: public eoInit<EOT>
{
    public:
        eoInitFixedLength(unsigned _howmany, Gen _generator = Gen()) 
            : howmany(_howmany), generator(_generator) {}

        void operator()(EOT& chrom)
        {
            chrom.resize(howmany);
            generate(chrom.begin(), chrom.end(), generator);
            chrom.invalidate();
        }

    private :
        unsigned howmany;
        Gen generator;
};

/**
    Initializor for variable length representations with a single type
*/
template <class EOT, class Gen>
class eoInitVariableLength: public eoInit<EOT>
{
    public:
        eoInitVariableLength(unsigned _minSize, unsigned _maxSize, Gen _generator = Gen()) 
            : offset(_minSize), extent(_maxSize - _minSize), generator(_generator) 
        {
            if (_minSize >= _maxSize)
                throw logic_error("eoInitVariableLength: minSize larger or equal to maxSize");
        }

        void operator()(EOT& chrom)
        {
            unsigned howmany = offset + rng.random(extent);
            chrom.resize(howmany);
            generate(chrom.begin(), chrom.end(), generator);
            chrom.invalidate();
        }

    private :
        unsigned offset;
        unsigned extent;
        Gen generator;
};


/**
    eoInitAdaptor changes the place in the hierarchy
    from eoInit to eoMonOp. This is mainly a type conversion,
    nothing else
    .
    @see eoInit, eoMonOp
*/
template <class EOT>
class eoInitAdaptor : public eoMonOp<EOT>
{
    public :
        eoInitAdaptor(eoInit<EOT>& _init) : init(_init) {}
    
        void operator()(EOT& _eot)
        {
            init(_eot);
        }
    private :
    
        eoInit<EOT>& init;
};

#endif
