//-----------------------------------------------------------------------------
// eoBin.h
//-----------------------------------------------------------------------------

#ifndef eoBin_h
#define eoBin_h

//-----------------------------------------------------------------------------

#include <iostream>     // ostream, istream
#include <function.h>   // bind2nd

#ifdef HAVE_BVECTOR_H
#include <bvector.h>
#error "incluyo bvector.h"
#elseif
#ifdef HAVE_VECTOR
#include <vector>
#define bit_vector vector<bool>
#error "incluyo vector"
#elseif
#error "are you kidding?"
#endif
#endif

#include <string>       // string
#include <EO.h>         // EO

//-----------------------------------------------------------------------------
/// eoBin: implementation of binary chromosome.
/// based on STL's bit_vector (vector<bool>)
//-----------------------------------------------------------------------------

template <class F> class eoBin: public EO<F>, public bit_vector
{
 public:
  typedef bool Type;

  /// (Default) Constructor.
  /// @param size Size of the binary string.
  eoBin(const unsigned& size = 0, const bool& value = false): 
    bit_vector(size, value) {}

  /// (Default) Constructor.
  /// @param size Size of the binary string.
  eoBin(const unsigned& size, const eoRnd<Type>& rnd): bit_vector(size) 
    {
      generate(begin(), end(), rnd);
    }
  
  /// Constructor from istream.
  /// @param is The istream to read from.
  eoBin(istrstream& is)
    { 
      readFrom(is); 
    }
  
  /// My class name.
  string className() const 
    { 
      return "eoBin"; 
    }
  
  /// To print me on a stream.
  /// @param os The ostream.
  void printOn(ostream& os) const
    {
      copy(begin(), end(), ostream_iterator<bool>(os));
    }
  
  /// To read me from a stream.
  /// @param is The istream.
  void readFrom(istream& is)
    {
      string bits;
      is >> bits;
      if (is)
	{
	  resize(bits.size());
	  transform(bits.begin(), bits.end(), begin(), 
		    bind2nd(equal_to<char>(), '1'));
	}
    }
};

//-----------------------------------------------------------------------------

#endif eoBin_h
