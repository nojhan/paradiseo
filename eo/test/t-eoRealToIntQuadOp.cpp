#include <iostream>

#include <eo>
#include <es.h>

using namespace std;

int main(int, char**)
{
    eoIntInterval intbounds(1,5);
    eoRealInterval rb(1,5);
    eoRealVectorBounds realbounds(5, rb);

    using Chrom = eoInt<double>;
    using CrossWrapper = eoRealToIntQuadOp<Chrom>;

    eoSegmentCrossover< typename CrossWrapper::EOTreal > crossreal(realbounds, /*alpha*/0);

    CrossWrapper crossint(crossreal, intbounds);

    Chrom sol1({1,2,3,4,5});
    Chrom sol2({1,2,3,4,5});

    bool changed = crossint(sol1, sol2);
    assert(changed);

    for(auto& x : sol1) {
        assert(intbounds.isInBounds(x));
    }
    for(auto& x : sol2) {
        assert(intbounds.isInBounds(x));
    }
}


