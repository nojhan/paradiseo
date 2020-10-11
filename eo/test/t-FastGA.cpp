
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
    size_t dim = 5;
    size_t pop_size = 3;

    oneMaxEval<EOT> evalfunc;
    eoPopLoopEval<EOT> eval(evalfunc);

    eoBooleanGenerator gen(0.5);
    eoInitFixedLength<EOT> init(dim, gen);

    double cross_rate = 0.5;
    eoProportionalOp<EOT> cross;
    // Cross-over that produce only one offspring,
    // made by wrapping the quad op (which produce 2 offsprings)
    // in a bin op (which ignore the second offspring).
    eo1PtBitXover<EOT> crossover;
    eoQuad2BinOp<EOT> mono_cross(crossover);
    cross.add(mono_cross, cross_rate);
    eoBinCloneOp<EOT> bin_clone;
    cross.add(bin_clone, 1 - cross_rate); // Clone

    double mut_rate = 0.5;
    eoProportionalOp<EOT> mut;
    eoShiftedBitMutation<EOT> mutation(0.5);
    mut.add(mutation, mut_rate);
    eoMonCloneOp<EOT> mon_clone;
    mut.add(mon_clone, 1 - mut_rate); // FIXME TBC

    eoSequentialOp<EOT> variator;
    variator.add(cross,1.0);
    variator.add(mut,1.0);

    double lambda = 1.0; // i.e. 100%
    eoStochTournamentSelect<EOT> selector(0.5);
    eoGeneralBreeder<EOT> breeder(selector, variator, lambda);

    eoGenContinue<EOT> common_cont(3);
    eoCombinedContinue<EOT> gen_cont(common_cont);
    //gen_cont.add(continuator);

    eoPlusReplacement<EOT> replacement;

    eoEasyEA<EOT> algo = eoEasyEA<EOT>(gen_cont, eval, breeder, replacement);

    eoPop<EOT> pop;
    pop.append(pop_size, init);
    eval(pop,pop);

    algo(pop);

    std::cout << pop.best_element() << std::endl;
}
