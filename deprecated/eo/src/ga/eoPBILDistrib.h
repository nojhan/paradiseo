// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoPBILDistrib.h
// (c) Marc Schoenauer, 2001
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

    Contact: Marc.Schoenauer@inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef _eoPBILDistrib_H
#define _eoPBILDistrib_H

#include <eoDistribution.h>

/**
 * Distribution Class for PBIL algorithm
 *      (Population-Based Incremental Learning, Baluja and Caruana 96)
 *
 * It encodes a univariate distribution on the space of bitstrings,
 * i.e. one probability for each bit to be one
 *
 * It is an eoValueParam<std::vector<double> > :
 *    the std::vector<double> stores the probabilities that each bit is 1
 *
 * It is still pure virtual, as the update method needs to be specified
*/

template <class EOT>
class eoPBILDistrib :  public eoDistribution<EOT>,
                       public eoValueParam<std::vector<double> >
{
public:
  /** Ctor with size of genomes, and update parameters */
  eoPBILDistrib(unsigned _genomeSize) :
    eoDistribution<EOT>(),
    eoValueParam<std::vector<double> >(std::vector<double>(_genomeSize, 0.5), "Distribution"),
    genomeSize(_genomeSize)
  {}

  /** the randomizer of indis */
  virtual void operator()(EOT & _eo)
  {
    _eo.resize(genomeSize);        // just in case
    for (unsigned i=0; i<genomeSize; i++)
      _eo[i] = eo::rng.flip(value()[i]);
    _eo.invalidate();              // DO NOT FORGET!!!
  }

  /** Accessor to the genome size */
  unsigned Size() {return genomeSize;}

  /** printing... */
  virtual void printOn(std::ostream& os) const
  {
    os << value().size() << ' ';
    for (unsigned i=0; i<value().size(); i++)
      os << value()[i] << ' ';
  }

  /** reading...*/
  virtual void readFrom(std::istream& is)
  {
    unsigned sz;
    is >> sz;

    value().resize(sz);
    unsigned i;

    for (i = 0; i < sz; ++i)
      {
        double atom;
        is >> atom;
        value()[i] = atom;
      }
  }

  unsigned int size() {return genomeSize;}

  virtual std::string className() const {return "eoPBILDistrib";};

private:
  unsigned genomeSize; // size of indis
};

#endif
