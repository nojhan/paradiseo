#include <iostream>

#include <eo>
#include <es.h>

using namespace std;

int main(int, char**)
{
    eoIntInterval bounds(1,5);

    using Chrom = eoInt<double>;
    using MutWrapper = eoRealToIntMonOp<Chrom>;

    eoDetUniformMutation< typename MutWrapper::EOTreal > mutreal(/*range*/6, /*nb*/5);

    MutWrapper mutint(mutreal, bounds);

    Chrom sol({1,2,3,4,5});

    bool changed = mutint(sol);
    assert(changed);

    for(auto& x : sol) {
        assert(bounds.isInBounds(x));
    }
}

