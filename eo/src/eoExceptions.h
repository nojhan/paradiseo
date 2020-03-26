/*
(c) Thales group, 2010

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

#ifndef __eoExceptions_h__
#define __eoExceptions_h__

#include <ctime>
#include <stdexcept>
#include <sstream>
#include <string>

//! You can catch this base exception if you want to catch anything thrown by ParadisEO. @ingroup Core
class eoException : public std::runtime_error
{
public:
    eoException(std::string msg = "") :
        std::runtime_error(""),
        _msg(msg)
    { }

    virtual std::string message() const throw()
    {
        return _msg;
    }

    const char* what() const throw()
    {
        return message().c_str();
    }

    ~eoException() throw() {}

protected:
    const std::string _msg;
};


class eoInvalidFitnessError : public eoException
{
public:
    eoInvalidFitnessError(std::string msg = "Invalid fitness") : eoException(msg) {}
    ~eoInvalidFitnessError() throw() {}
};


class eoPopSizeException : public eoException
{
public:
    eoPopSizeException(size_t size, std::string msg = "") :
        eoException(),
        _size(size),
        _msg(msg)
    {}

    virtual std::string message() const throw()
    {
        std::ostringstream oss;
        oss << "Bad population size: " << _size;
        if(_msg != "") {
            oss << ", " << _msg;
        }
        return oss.str();
    }

    ~eoPopSizeException() throw() {}
protected:
    const size_t _size;
    const std::string _msg;
};


class eoPopSizeChangeException : public eoPopSizeException
{
public:
    eoPopSizeChangeException(size_t size_from, size_t size_to, std::string msg="") :
        eoPopSizeException(0),
        _size_from(size_from),
        _size_to(size_to),
        _msg(msg)
    {}

    virtual std::string message() const throw()
    {
        std::ostringstream oss;
        oss << "Population size changed from " << _size_from << " to " << _size_to;
        if(_msg != "") {
            oss << ", " << _msg;
        }
        return oss.str();
    }

    ~eoPopSizeChangeException() throw() {}
protected:
    const size_t _size_from;
    const size_t _size_to;
    const std::string _msg;
};


/** Base class for exceptions which need to stop the algorithm to be handled
 * 
 * (like stopping criterion or numerical errors).
 */
class eoStopException : public eoException
{
public:
    eoStopException(std::string msg = "") : eoException(msg) {}
    ~eoStopException() throw() {}
};


//! Base class for limit-based exceptions (see eoMaxTimeException and eoMaxEvalException.
class eoMaxException : public eoStopException
{
public:
    eoMaxException(std::string msg = "") : eoStopException(msg) {}
     ~eoMaxException() throw() {}
};


/*!
An error that signals that some bad data have been returned.

Thrown by @see eoEvalNanThrowException

@ingroup Evaluation
*/
class eoNanException : public eoStopException
{
public:
    eoNanException() :
        eoStopException("The objective function returned a bad value (nan or inf)")
    { }
    ~eoNanException() throw() {}
};


/*!
An error that signals that a maximum elapsed time has been reached.

Thrown by @see eoEvalTimeThrowException

@ingroup Evaluation
*/
class eoMaxTimeException : public eoMaxException
{
public:
    eoMaxTimeException( time_t elapsed) :
        eoMaxException(),
        _elapsed(elapsed)
    { }

    virtual std::string message() const throw()
    {
        std::ostringstream msg;
        msg << "The maximum number of allowed seconds has been reached ("
             << _elapsed << ")";
        return msg.str();
    }

    ~eoMaxTimeException() throw() {}

protected:
    const time_t _elapsed;
};


/*!
An error that signals that a maximum number of evaluations has been reached.

Thrown by @see eoEvalThrowException

@ingroup Evaluation
*/
class eoMaxEvalException : public eoMaxException
{
public:
    eoMaxEvalException(unsigned long threshold) :
        eoMaxException(),
        _threshold(threshold)
    { }

    virtual std::string message() const throw()
    {
        std::ostringstream msg;
        msg << "The maximum number of evaluation has been reached ("
             << _threshold << ").";
        return msg.str();
    }

    ~eoMaxEvalException() throw() {}

protected:
    const unsigned long _threshold;
};

//! Base class for exceptions related to eoParam management. @ingroup Parameters
class eoParamException : public eoException
{
public:
    eoParamException(std::string msg = "") : eoException(msg) {}
};

/*!
 * An error that signals a missing parameter
 *
 * Thrown by eoParser::getParam
 *
 * @ingroup Parameters
 */
class eoMissingParamException : public eoParamException
{
public:
    eoMissingParamException(std::string name) :
        eoParamException(),
        _name(name)
    { }

    virtual std::string message() const throw()
    {
        std::ostringstream msg;
        msg << "The command parameter " << _name << " has not been declared";
        return msg.str();
    }

    ~eoMissingParamException() throw() {}

protected:
    const std::string _name;
};


/*!
 * An error that signals a bad parameter type
 *
 * Thrown by eoParser::valueOf
 *
 * @ingroup Parameters
 */
class eoWrongParamTypeException : public eoParamException
{
public:
    eoWrongParamTypeException(std::string name) :
        eoParamException(),
        _name(name)
    { }

    virtual std::string message() const throw()
    {
        std::ostringstream msg;
        msg << "You asked for the parameter " << _name
             << " but it has not been declared under this type";
        return msg.str();
    }

    ~eoWrongParamTypeException() throw() {}

protected:
    const std::string _name;
};


//! Exception related to a system call.
class eoSystemError : public eoException
{
public:
    eoSystemError(std::string cmd) :
        eoException(),
        _cmd(cmd), _has_pipe(false), _err_code(-1), _output("")
    { }

    eoSystemError(std::string cmd, int err_code, std::string output = "") :
        eoException(),
        _cmd(cmd), _has_pipe(true), _err_code(err_code), _output(output)
    { }

    virtual std::string message() const throw()
    {
        std::ostringstream ss;
        ss << "System call: `" << _cmd << "` ended with error";
        if(_has_pipe) {
            ss << " code #" << _err_code;
            if(_output != "") {
               ss << " with the following output:" << std::endl << _output;
            }
        }
        return ss.str();
    }

    ~eoSystemError() throw() {}

protected:
    const std::string _cmd;
    const bool _has_pipe;
    const int _err_code;
    const std::string _output;
};


class eoFileError : public eoSystemError
{
public:
    eoFileError(std::string filename) :
        eoSystemError(""),
        _filename(filename)
    { }

    virtual std::string message() const throw()
    {
        std::ostringstream oss;
        oss << "Could not open file: " << _filename;
        return oss.str();
    }

protected:
    std::string _filename;
};

#endif // __eoExceptions_h__
