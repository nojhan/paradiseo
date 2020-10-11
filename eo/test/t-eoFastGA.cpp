
/*
   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation;
   version 2 of the License.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include <iostream>
#include <string>

#include <eo>
#include <ga.h>
#include "../../problems/eval/oneMaxEval.h"

using EOT = eoBit<double>;

int main(int /*argc*/, char** /*argv*/)
{
    size_t dim = 100;
    size_t pop_size = 10;

    oneMaxEval<EOT> evalfunc;
    eoPopLoopEval<EOT> eval(evalfunc);

    eoBooleanGenerator gen(0.5);
    eoInitFixedLength<EOT> init(dim, gen);

    double cross_rate = 0.5;
    eoSequentialSelect<EOT> select_cross;
    eoUBitXover<EOT> crossover;
    eoRandomSelect<EOT> select_aftercross;

    double mut_rate = 0.5;
    eoSequentialSelect<EOT> select_mut;
    eoStandardBitMutation<EOT> mutation(0.5);

    eoPlusReplacement<EOT> replacement;

    eoGenContinue<EOT> common_cont(dim*2);
    eoCombinedContinue<EOT> gen_cont(common_cont);

    eoFastGA<EOT> algo(
        cross_rate,
        select_cross,
        crossover,
        select_aftercross,
        mut_rate,
        select_mut,
        mutation,
        eval,
        replacement,
        common_cont
    );

    eoPop<EOT> pop;
    pop.append(pop_size, init);
    eval(pop,pop);

    algo(pop);

    std::cout << pop.best_element() << std::endl;
}
