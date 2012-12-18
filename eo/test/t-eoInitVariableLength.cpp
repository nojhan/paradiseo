#include <iostream>
#include <vector>
#include <eo>

// An adhoc atom type of our own
class Quad : public std::vector<int>
{
    public:
        // Just four times zero
        Quad() : std::vector<int>(4,0) {}
};

// EO somewhat forces you to implement a way to read/print your atom type
// You can either inherit from eoPrintable and overload readFrom/printOn
// or, just like here, directly overload stream operators.

// read
std::istream& operator>>( std::istream& is, Quad& q )
{
    for( unsigned int i=0, n=4; i<n; ++i) {
        // use default int stream input
        is >> q[i];
    }
    return is;
}

// print
std::ostream& operator<<( std::ostream& os, const Quad& q )
{
    os << q[0];
    for( unsigned int i=1, n=4; i<n; ++i) {
        os << " " << q[i];
    }
    os << "  ";
    return os;
}

// An init for the atoms
// Note that this mask the template passed to the eoInit
class QuadInit : public eoInit<Quad>
{
    public:
        // this is the API: an init modify the solution
        void operator()( Quad& q ) {
            for( unsigned int i=0, n=4; i<n; ++i) {
                // rng is the random number generator of EO
                q[i] = rng.random(10);
            }
        }
};

// The solution/individual type.
// Just a proxy to an eoVector of atoms,
// with a fitness as double.
class QuadVec : public eoVector<double,Quad>
{};


int main()
{
    unsigned int vec_size_min = 1;
    unsigned int vec_size_max = 10;
    unsigned int pop_size = 10;

    // Fix a seed for the random generator,
    // thus, the results are predictable.
    // Set it to zero if you want pseudo-random numbers
    // that changes at each calls.
    rng.reseed( 1 );

    // The operator that produce a random vector of four values.
    QuadInit atom_init;

    // The operator that produces a random vector of a (vector of four values).
    eoInitVariableLength<QuadVec> vec_init( vec_size_min, vec_size_max, atom_init );

    // You can initialize a population of N individuals by passing an initializer to it.
    eoPop<QuadVec> pop( pop_size, vec_init );

    // eoPop can be printed easily,
    // thanks to the overloadings above.
    std::cout << pop << std::endl;

// With a seed at 1, this should output:
/*
10
INVALID  6 5 9 5 9   0 1 6 0   4 8 9 0   6 9 4 9   5 5 3 6   3 0 2 8   
INVALID  9 9 2 0 3   2 4 3 3   6 2 8 2   4 5 4 7   5 3 0 5   4 9 8 3   2 7 7 9   4 4 4 6   6 3 9 2   
INVALID  1 1 4 1 4   
INVALID  5 3 8 9 8   8 1 4 1   6 6 5 4   3 2 7 5   1 2 6 1   
INVALID  3 7 8 1 4   0 9 1 0   6 4 2 1   
INVALID  6 7 4 6 8   1 2 6 0   5 1 2 6   9 2 6 8   6 1 5 5   4 1 0 3   
INVALID  5 2 7 7 6   1 4 0 7   5 5 9 7   2 4 7 1   6 1 9 0   
INVALID  3 5 5 3 9   2 9 9 1   1 7 2 1   
INVALID  6 9 9 9 0   0 7 1 7   9 7 8 5   3 7 5 6   7 3 6 7   6 3 3 5   
INVALID  1 6 2 4 3   
*/
}
