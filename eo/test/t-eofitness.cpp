//-----------------------------------------------------------------------------
// t-eofitness.cpp
// (c) GeNeura Team 1998
//-----------------------------------------------------------------------------

#include <iostream>  // cout
#include <eo>        // eoFitness

//-----------------------------------------------------------------------------

class eoFloat: public eoFitness
{
public:
  bool operator<(const eoFitness& other) const
    {
      const eoFloat& x = (const eoFloat&) other;
      return fitness < x.fitness;
    }

  operator float() const
    {
      return fitness;
    }

  void printOn(ostream& os) const
    {
      os << fitness;
    }
  
  void readFrom(istream& is)
    {
      is >> fitness;
    }

private:
  float fitness;
};

//-----------------------------------------------------------------------------

main()
{
  eoFloat a, b;

  unsigned repeat = 2;
  while (repeat--)
    {
      cout << "------------------------------------------------------" << endl;
      cout << "testing <    ";
      if (a < b)
	cout << a << " < " << b << "  is true" << endl;
      else
	cout << a << " < " << b << "  is false" <<endl;
      
      cout << "testing >    ";
      if (a > b)
	cout << a << " > " << b << "  is true" << endl;
      else
	cout << a << " > " << b << "  is false" <<endl;
      
      cout << "testing ==   ";
      if (a == b)
	cout << a << " == " << b << " is true" << endl;
      else
	cout << a << " == " << b << " is false" <<endl; 
      
      cout << "testing !=   ";
      if (a != b)
	cout << a << " != " << b << " is true" << endl;
      else
	cout << a << " != " << b << " is false" <<endl;
      
      a = b;
    }
}

//-----------------------------------------------------------------------------
