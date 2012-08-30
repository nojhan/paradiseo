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

#include <algorithm>

#include <eoOp.h>
#include <eoSTLFunctor.h>
#include <utils/eoRndGenerators.h>
#include <utils/rnd_generators.h>  // for shuffle method


/**
    @defgroup Initializators Initialization operators
    @ingroup Operators

    Initializators are operators that creates initial individuals and populations.
*/
/** @{*/
/**
        Base (name) class for Initialization of chromosomes, used in a population
        contructor. It is derived from eoMonOp, so it can be used
    inside the algorithm as well.

        @see eoPop
*/
template <class EOT>
class eoInit : public eoUF<EOT&, void>
{
public:

  /** className: Mandatory because of eoCombinedInit.
     SHould be pure virtual, but then we should go over the whole
   * code to write the method for all derived classes ... MS 16/7/04 */
  virtual std::string className(void) const { return "eoInit"; }
};

/** turning an eoInit into a generator
 * probably we should only use genrators - and suppress eoInit ???
 * MS - July 2001
 */
template <class EOT>
class eoInitGenerator :  public eoF<EOT>
{
public:

  /** Ctor from a plain eoInit */
  eoInitGenerator(eoInit<EOT> & _init):init(_init) {}

  virtual EOT operator()()
    {
      EOT p;
      init(p);
      return (p);
    }
private:
  eoInit<EOT> & init;
};

/**
    Initializer for fixed length representations with a single type
*/
template <class EOT>
class eoInitFixedLength: public eoInit<EOT>
{
    public:

    typedef typename EOT::AtomType AtomType;

        eoInitFixedLength(unsigned _combien, eoRndGenerator<AtomType>& _generator)
            : combien(_combien), generator(_generator) {}

        virtual void operator()(EOT& chrom)
        {
            chrom.resize(combien);
            std::generate(chrom.begin(), chrom.end(), generator);
            chrom.invalidate();
        }

    private :
        unsigned combien;
        /// generic wrapper for eoFunctor (s), to make them have the function-pointer style copy semantics
        eoSTLF<AtomType> generator;
};

/**
    Initializer for variable length representations with a single type
*/
template <class EOT>
class eoInitVariableLength: public eoInit<EOT>
{
public:
typedef typename EOT::AtomType AtomType;

//   /** Ctor from a generator */
//   eoInitVariableLength(unsigned _minSize, unsigned _maxSize, eoF<typename EOT::AtomType> & _generator = Gen())
//     : offset(_minSize), extent(_maxSize - _minSize),
//                       repGenerator( eoInitGenerator<typename EOT::AtomType>(*(new eoInit<EOT>)) ),
//                       generator(_generator)
//   {
//     if (_minSize >= _maxSize)
//       throw std::logic_error("eoInitVariableLength: minSize larger or equal to maxSize");
//   }

  /** Ctor from an eoInit */
  eoInitVariableLength(unsigned _minSize, unsigned _maxSize, eoInit<AtomType> & _init)
    : offset(_minSize), extent(_maxSize - _minSize), init(_init)
  {
    if (_minSize >= _maxSize)
      throw std::logic_error("eoInitVariableLength: minSize larger or equal to maxSize");
  }


  virtual void operator()(EOT& _chrom)
  {
    _chrom.resize(offset + rng.random(extent));
    typename std::vector<AtomType>::iterator it;
    for (it=_chrom.begin(); it<_chrom.end(); it++)
      init(*it);
    _chrom.invalidate();
  }

  // accessor to the atom initializer (needed by operator constructs sometimes)
  eoInit<AtomType> & atomInit() {return init;}

private :
  unsigned offset;
  unsigned extent;
  eoInit<AtomType> & init;
};


/**
    Initializer for permutation (integer-based) representations.
*/
template <class EOT>
class eoInitPermutation: public eoInit<EOT>
{
    public:

    typedef typename EOT::AtomType AtomType;

        eoInitPermutation(unsigned _chromSize, unsigned _startFrom=0)
            : chromSize(_chromSize), startFrom(_startFrom){}

        virtual void operator()(EOT& chrom)
        {
            chrom.resize(chromSize);
            for(unsigned idx=0;idx <chrom.size();idx++)
                        chrom[idx]=idx+startFrom;

            std::random_shuffle(chrom.begin(), chrom.end(),gen);
            chrom.invalidate();
        }

    private :
        unsigned chromSize;
        unsigned startFrom;
        UF_random_generator<unsigned int> gen;
};
/** @example t-eoInitPermutation.cpp
 */


/**
    eoInitAdaptor changes the place in the hierarchy
    from eoInit to eoMonOp. This is mainly a type conversion,
    nothing else

    @see eoInit, eoMonOp
*/
template <class EOT>
class eoInitAdaptor : public eoMonOp<EOT>
{
    public :
        eoInitAdaptor(eoInit<EOT>& _init) : init(_init) {}

        bool operator()(EOT& _eot)
        {
            init(_eot);
            return true;
        }
    private :

        eoInit<EOT>& init;
};

#endif
/** @}*/
