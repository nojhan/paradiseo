// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoCombinedInit.h
// (c) Maarten Keijzer, GeNeura Team, Marc Schoenauer 2004
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

#ifndef _eoCombinedInit_h
#define _eoCombinedInit_h

#include <eoInit.h>

/**
    Combined INIT: a proportional recombination of eoInit objects

    @ingroup Initializators
*/
template< class EOT>
class eoCombinedInit: public eoInit<EOT> {
public:

  /** Ctor, make sure that at least one eoInit is present */
  eoCombinedInit( eoInit<EOT>& _init, double _rate)
    : eoInit<EOT> ()
  {
    initializers.push_back(&_init);
    rates.push_back(_rate);
  }

  /* FIXME remove in next release
  void add(eoInit<EOT> & _init, double _rate, bool _verbose)
  {
      eo::log << eo::warnings << "WARNING: the use of the verbose parameter in eoCombinedInit::add is deprecated and will be removed in the next release." << std::endl;
      add( _init, _rate );
  }
  */

  /** The usual method to add objects to the combination
   */
  void add(eoInit<EOT> & _init, double _rate)
  {
    initializers.push_back(&_init);
    rates.push_back(_rate);
    // compute the relative rates in percent - to warn the user!
      printOn( eo::log << eo::logging );
  }

  /** outputs the operators and percentages */
  virtual void printOn(std::ostream & _os)
  {
    double total = 0;
    unsigned i;
    for (i=0; i<initializers.size(); i++)
      total += rates[i];
    _os << "In " << className() << "\n" ;
    for (i=0; i<initializers.size(); i++)
      _os << initializers[i]->className() << " with rate " << 100*rates[i]/total << " %\n";
  }

  /** Performs the init: chooses among all initializers
   * using roulette wheel on the rates
   */
  virtual void operator() ( EOT & _eo )
  {
    unsigned what = rng.roulette_wheel(rates); // choose one op
    (*initializers[what])(_eo);            // apply it
    return;
  }

  virtual std::string className(void) const { return "eoCombinedInit"; }

private:
std::vector<eoInit<EOT>*> initializers;
std::vector<double> rates;
};

#endif
