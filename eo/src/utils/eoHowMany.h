// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoHowMany_h.h
//   Base class for choosing a number of guys to apply something from a popsize
// (c) Marc Schoenauer, 2000
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

#ifndef eoHowMany_h
#define eoHowMany_h

/**
 * to be used in selection / replacement procedures to indicate whether 
 * the argument (rate, a double) shoudl be treated as a rate (number=rate*popSize)
 * or as an absolute integer (number=rate regardless of popsize).
 * the default value shoudl ALWAYS be true (eo_as_a_rate).
 *
 * this construct is mandatory because in some cases you might not know the 
 * population size that will enter the replacement. For instance, you 
 * cannot simply have a pre-computed (double) rate of 1/popSize 
 * if you want 1 guy
 *
 * Example use: in <a href="class_eogeneralbreeder.html">eoGeneralBreeder.h</a>
 * Example reading from parser: in 
 *         <a href="make_algo_scalar_h-source.html">do/make_algo_scalar.h line 141</a>
 */

class eoHowMany : public eoPersistent
{
public:
  /** Original Ctor from direct rate + bool */
  eoHowMany(double  _rate = 0.0, bool _interpret_as_rate = true):
    rate(0), combien(0)
  {
    if (_interpret_as_rate)
      {
	rate = _rate;
      }
    else
      {
	if (_rate<0)
	  throw std::logic_error("Negative number in eoHowMany!");
	combien = (unsigned int)_rate;
	if (combien != _rate)
	  cout << "Warning: Number was rounded in eoHowMany";
      }
  }

  /// Virtual dtor. They are needed in virtual class hierarchies.
  virtual ~eoHowMany() {}

  unsigned int operator()(unsigned int _size)
  {
    if (combien == 0)
      {
	return (unsigned int) (rate * _size);
      }
    return combien;
  }

  virtual void printOn(ostream& _os) const 
  {
    if (combien == 0)
      _os << 100*rate << "% " << std::ends;
    else
      _os << combien << " " << std::ends;
    return;

  }

  virtual void readFrom(istream& _is) 
  {
    string value;
    _is >> value;
    readFrom(value);
    return;
  }

  void readFrom(std::string _value)
  {
    // check for %
    bool interpret_as_rate = false;   // == no %
    size_t pos =  _value.find('%');
    if (pos < _value.size())  //  found a %
      {
	interpret_as_rate = true;
	_value.resize(pos);	   // get rid of %
      }
    std::istrstream is(_value.c_str());
    is >> rate;
    // now store
    if (interpret_as_rate)
      {
	combien = 0;
	rate /= 100.0;
      }
    else
      combien = unsigned(rate);	   // and rate will not be used

    // minimal check
    if ( (combien <= 0) && (rate <= 0.0) )
      throw runtime_error("Invalid parameters read in eoHowMany::readFrom");
  }
  
private :
  double rate;
  unsigned combien;
};



#endif
