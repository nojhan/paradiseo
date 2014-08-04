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

class eoMaxException : public std::exception {};



/*!
An error that signals that a maximum elapsed time has been reached.

Thrown by @see eoEvalTimeThrowException

@ingroup Evaluation
*/
class eoMaxTimeException : public eoMaxException
{
public:
    eoMaxTimeException( time_t elapsed) : _elapsed(elapsed) {}

    virtual const char* what() const throw()
    {
        std::ostringstream ss;
        ss << "STOP in eoMaxTimeException: the maximum number of allowed seconds has been reached (" << _elapsed << ").";
        return ss.str().c_str();
    }

private:
    time_t _elapsed;
};


/*!
An error that signals that a maximum number of evaluations has been reached.

Thrown by @see eoEvalEvalThrowException

@ingroup Evaluation
*/
class eoMaxEvalException : public eoMaxException
{
public:
    eoMaxEvalException(unsigned long threshold) : _threshold(threshold){}

    virtual const char* what() const throw()
    {
        std::ostringstream ss;
        ss << "STOP in eoMaxEvalException: the maximum number of evaluation has been reached (" << _threshold << ").";
        return ss.str().c_str();
    }

private:
    unsigned long _threshold;
};


/*!
 * An error that signals a missing parameter
 *
 * Thrown by eoParser::getParam
 *
 * @ingroup Parameters 
 */
class eoMissingParamException : public std::exception
{
public:
    eoMissingParamException(std::string name) : _name(name){}

    virtual const char* what() const throw()
    {
        std::ostringstream ss;
        ss << "The command parameter " << _name << " has not been declared";
        return ss.str().c_str();
    }

    ~eoMissingParamException() throw() {}

private:
    std::string _name;
};

/*!
 * An error that signals a bad parameter type
 *
 * Thrown by eoParser::valueOf
 *
 * @ingroup Parameters 
 */
class eoWrongParamTypeException : public std::exception
{
public:
    eoWrongParamTypeException(std::string name) : _name(name){}

    virtual const char* what() const throw()
    {
        std::ostringstream ss;
        ss << "You asked for the parameter " << _name << " but it has not been declared under this type";
        return ss.str().c_str();
    }

    ~eoWrongParamTypeException() throw() {}

private:
    std::string _name;
};

#endif // __eoExceptions_h__
