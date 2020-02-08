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

#ifndef eoEvalCmd_H
#define eoEvalCmd_H

#include <cstdlib>
#include <array>

#include "eoEvalFunc.h"
#include "eoExceptions.h"


/** Call a system command to evaluate an individual.
 *
 * @note Tested only under Unix systems, may not be portable as is.
 *
 * Use the default string serialization of the EOT
 * and pass it as command line arguments of the command.
 * The command is expected to output on stdout a string
 * that can be interpreted as a float by `atof`.
 *
 * @todo Use the serialization of fitness instead of atof.
 *
 * For example, an eoReal would lead to a call like:
 * ```>&2 cmd INVALID 3 1.0 2.1 3.2```
 *
 * Throw an eoSystemError exception if the command exited with
 * a return code different from zero.
 *
 *@ingroup Evaluation
 */
template<class EOT, int BUFFER_SIZE = 128>
class eoEvalCmd : public eoEvalFunc< EOT >
{
public:
    using Fitness = typename EOT::Fitness;

    /** Constructor
     *
     * @note The prefix and suffix are automatically
     * separated from the command by a space.
     *
     * The formated command looks like: `prefix cmd infix sol suffix`
     *
     * The default prefix allows to redirect any output to stdout under Unix.
     *
     * @param cmd The command to run.
     * @param prefix Inserted before the command
     * @param suffix Inserted between cmd and the serialized solution.
     * @param suffix Append after the solution.
     */
    eoEvalCmd( const std::string cmd,
               const std::string prefix = ">&1",
               const std::string infix = "",
               const std::string suffix = ""
        ) :
        _cmd(cmd),
        _suffix(suffix),
        _infix(infix),
        _prefix(prefix),
        _last_call("")
    {}

    virtual void operator()( EOT& sol )
    {
        // Any modification to sol would makes it invalid,
        // it is thus useless to evaluate it, if it is not invalid.
        if(not sol.invalid()) {
            return;
        }

        sol.fitness( call( sol ) );
    }

    //! Return the last command string that was called.
    std::string last_call() const
    {
        return _last_call;
    }

private:
    const std::string _cmd;
    const std::string _prefix;
    const std::string _infix;
    const std::string _suffix;
    std::string _last_call;

    Fitness call( EOT& sol )
    {
        std::array<char, BUFFER_SIZE> buffer;
        std::string result;

        std::ostringstream cmd;

        cmd << _prefix << " " << _cmd << " "
            << _infix  << " " <<  sol << " " << _suffix;

        // Keep track of the built command for debugging purpose.
        _last_call = cmd.str();

        FILE* pipe = popen(cmd.str().c_str(), "r");
        if(not pipe) {
            throw eoSystemError(cmd.str());
        }
        while(fgets(buffer.data(), BUFFER_SIZE, pipe) != NULL) {
            result += buffer.data();
        }
        auto return_code = pclose(pipe);

        if(return_code != 0) {
            throw eoSystemError(cmd.str(), return_code, result);
        }

        // FIXME Use serialized input for the fitness instead of atof.
        Fitness f = std::atof(result.c_str());

        return f;
    }
};

#endif // eoEvalCmd_H
