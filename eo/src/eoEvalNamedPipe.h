/*
(c) Thales group, 2020

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
Johann Dréo <johann.dreo@thalesgroup.com>
*/

#ifndef eoEvalNamedPipe_H
#define eoEvalNamedPipe_H

#include <fstream>
#include <sstream>
#include <cstdlib>
#include <array>

#include "eoEvalFunc.h"
#include "eoExceptions.h"

/** Communicate through a named pipe FIFO to evaluate an individual.
 *
 * With this class, you can plug an external process computing the fitness
 * of a serialized solution with a process hosting the EO algorithm.
 * Both processes just have to read/write solutions/fitness in files.
 *
 * The code would work with any file, but it is actually only useful with
 * FIFO pipes, which are blocking on I/O.
 * Thus, the process actually computing the fitness will ait for the solution to be wrote,
 * then compute and write the fitness back, waiting it to be read.
 * Conversely, the EO process will wait after having wrote the solution, that the other process
 * actually read it, then wait itself for the fitness to be read in the pipe.
 * With pipes, the synchronization of the two processes is guaranteed.
 *
 * To create a named FIFO pipe under Linux, see the command `mkfifo`.
 *
 * @note: if you use a single pipe for input/output, take care
 * of the synchronization with the process handling the fitness computation.
 * In particular, the first call of eoEvalNamedPipe
 * is to write the solution, THEN to read the fitness.
 *
 * @note Tested only under Unix systems, may not be portable as is.
 *
 * Use the default string serialization of the EOT and
 * the default deserialization of the fitness.
 *
 *@ingroup Evaluation
 */
template<class EOT>
class eoEvalNamedPipe : public eoEvalFunc< EOT >
{
public:
    using Fitness = typename EOT::Fitness;

    /** Constructor
     *
     * @param output_pipe_name The named pipe in which to write the serialized solution.
     * @param input_pipe_name The named pipe in which to read the serialized fitness. If it is "", use the output pipe.
     */
    eoEvalNamedPipe(
            const std::string output_pipe_name,
            const std::string input_pipe_name = ""
        ) :
        _output_pipe_name(output_pipe_name),
        _input_pipe_name(input_pipe_name)
    {
        if( _input_pipe_name == "") {
            _input_pipe_name = _output_pipe_name;
        }
    }

    virtual void operator()( EOT& sol )
    {
        // Any modification to sol would makes it invalid,
        // it is thus useless to evaluate it, if it is not invalid.
        if(not sol.invalid()) {
            return;
        }

        sol.fitness( call( sol ) );
    }

private:
    const std::string _output_pipe_name;
    std::string _input_pipe_name;

    Fitness call( EOT& sol )
    {
        // Write the passed solution.
        std::ofstream out(_output_pipe_name);
        out << sol << std::endl;
        out.close();

        // Read the output string in a valid fitness object.
        Fitness fit;
        std::ifstream if_fit(_input_pipe_name);
        std::stringstream ss_fit;
        ss_fit << if_fit.rdbuf();
        if_fit.close();
        ss_fit >> fit;

        return fit;
    }
};

#endif // eoEvalNamedPipe_H
