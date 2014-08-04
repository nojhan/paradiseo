/*
(c) Thales group, 2012

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
Contact: http://eodev.sourceforge.net

Authors:
    Benjamin Bouvier <benjamin.bouvier@gmail.com>
*/

/*
 * This file shows an example of parallel evaluation of a population, when using an eoEasyEA algorithm.
 * Moreover, we add a basic wrapper on the parallel evaluation, so as to show how to retrieve the best solutions.
 */
//-----------------------------------------------------------------------------

#include <eo>
#include <eoPopEvalFunc.h>

#include <es/make_real.h>
#include "../../eo/test/real_value.h"

#include <eoMpi.h>

#include <vector>
using namespace std;

//-----------------------------------------------------------------------------

class eoRealSerializable : public eoReal< eoMinimizingFitness >, public eoserial::Persistent
{
    public:

        eoRealSerializable(unsigned size = 0, double value = 0.0):
            eoReal<eoMinimizingFitness>(size, value) {}

        eoserial::Object* pack() const
        {
            eoserial::Object* obj = new eoserial::Object;
            obj->add( "array",
                    eoserial::makeArray< vector<double>, eoserial::MakeAlgorithm >
                    ( *this )
                    );

            bool invalidFitness = invalid();
            obj->add("invalid", eoserial::make( invalidFitness ) );
            if( !invalidFitness )
            {
                double fitnessVal = fitness();
                obj->add("fitness", eoserial::make( fitnessVal ) );
            }
            return obj;
        }

        void unpack( const eoserial::Object* obj )
        {
            this->clear();
            eoserial::unpackArray< vector<double>, eoserial::Array::UnpackAlgorithm >
                ( *obj, "array", *this );

            bool invalidFitness;
            eoserial::unpack( *obj, "invalid", invalidFitness );
            if( invalidFitness ) {
                invalidate();
            } else {
                double fitnessVal;
                eoserial::unpack<double>( *obj, "fitness", fitnessVal );
                fitness( fitnessVal );
            }
        }
};

typedef eoRealSerializable EOT;

/*
 * Wrapper for HandleResponse: shows the best answer, as it is found.
 *
 * Finding the best solution is an associative operation (as it is based on a "min" function, which is associative too)
 * and that's why we can perform it here. Indeed, the min element of 5 elements is the min element of the 3 first
 * elements and the min element of the 2 last elements:
 * min(1, 2, 3, 4, 5) = min( min(1, 2, 3), min(4, 5) )
 *
 * This is a reduction. See MapReduce example to have another examples of reduction.
 */
struct CatBestAnswers : public eo::mpi::HandleResponseParallelApply<EOT>
{
    CatBestAnswers()
    {
        best.fitness( 1000000000. );
    }

    /*
        our structure inherits the member _wrapped from HandleResponseFunction,
        which is a HandleResponseFunction pointer;

        it inherits also the member _d (like Data), which is a pointer to the
        ParallelApplyData used in the HandleResponseParallelApply&lt;EOT&gt;. Details
        of this data are contained in the file eoParallelApply. We need just to know that
        it contains a member assignedTasks which maps a worker rank and the sent slice
        to be processed by the worker, and a reference to the processed table via the
        call of the data() function.
    */

    // if EOT were a template, we would have to do: (thank you C++ :)
    // using eo::mpi::HandleResponseParallelApply<EOT>::_wrapped;
    // using eo::mpi::HandleResponseParallelApply<EOT>::d;

    void operator()(int wrkRank)
    {
        eo::mpi::ParallelApplyData<EOT> * d = _data;
        // Retrieve informations about the slice processed by the worker
        int index = d->assignedTasks[wrkRank].index;
        int size = d->assignedTasks[wrkRank].size;
        // call to the wrapped function HERE
        (*_wrapped)( wrkRank );
        // Compare fitnesses of evaluated individuals with the best saved
        for(int i = index; i < index+size; ++i)
        {
            if( best.fitness() < d->table()[ i ].fitness() )
            {
                eo::log << eo::quiet << "Better solution found:" << d->table()[i].fitness() << std::endl;
                best = d->table()[ i ];
            }
        }
    }

    protected:

    EOT best;
};

int main(int ac, char** av)
{
    eo::mpi::Node::init( ac, av );
    // eo::log << eo::setlevel( eo::debug );
    eo::log << eo::setlevel( eo::quiet );

    eoParser parser(ac, av);

    unsigned int popSize = parser.getORcreateParam((unsigned int)100, "popSize", "Population Size", 'P', "Evolution Engine").value();
    unsigned int dimSize = parser.getORcreateParam((unsigned int)10, "dimSize", "Dimension Size", 'd', "Evolution Engine").value();

    uint32_t seedParam = parser.getORcreateParam((uint32_t)0, "seed", "Random number seed", 0).value();
    if (seedParam == 0) { seedParam = time(0); }

    make_parallel(parser);
    make_help(parser);

    rng.reseed( seedParam );

    eoUniformGenerator< double > gen(-5, 5);
    eoInitFixedLength< EOT > init( dimSize, gen );

    eoEvalFuncPtr< EOT, double, const std::vector< double >& > mainEval( real_value );
    eoEvalFuncCounter< EOT > eval( mainEval );

    // until this point, everything (but eo::mpi::Node::init) is exactly as in an sequential version.
    // We then instanciate the parallel algorithm. The store is directly used by the eoParallelPopLoopEval, which
    // internally uses parallel apply.
    int rank = eo::mpi::Node::comm().rank();
    eo::mpi::DynamicAssignmentAlgorithm assign;
    if( rank == eo::mpi::DEFAULT_MASTER )
    {
        eoPop< EOT > pop( popSize, init );

        eo::log << "Size of population : " << popSize << std::endl;

        eo::mpi::ParallelApplyStore< EOT > store( eval, eo::mpi::DEFAULT_MASTER );
        store.wrapHandleResponse( new CatBestAnswers );

        eoParallelPopLoopEval< EOT > popEval( assign, eo::mpi::DEFAULT_MASTER, &store );

        //eoParallelPopLoopEval< EOT > popEval( assign, eo::mpi::DEFAULT_MASTER, eval, 5 );

        eo::log << eo::quiet << "Before first evaluation." << std::endl;
        popEval( pop, pop );
        eo::log << eo::quiet << "After first evaluation." << std::endl;

        pop = eoPop< EOT >( popSize, init );
        popEval( pop, pop );
        eo::log << eo::quiet << "After second evaluation." << std::endl;

        eo::log << eo::quiet << "DONE!" << std::endl;
    } else
    {
        eoPop< EOT > pop; // the population doesn't have to be initialized, as it is not used by workers.
        eoParallelPopLoopEval< EOT > popEval( assign, eo::mpi::DEFAULT_MASTER, eval );
        popEval( pop, pop );
    }

    return 0;
}

//-----------------------------------------------------------------------------
