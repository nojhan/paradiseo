#ifndef eoEvalFuncCounterBounder_H
#define eoEvalFuncCounterBounder_H

#include <eoEvalFunc.h>
#include <utils/eoParam.h>

/** @addtogroup Evaluation
 * @{
 */

/** The exception raised by eoEvalFuncCounterBounder
 * when the maximum number of allowed evaluations is reached.
 */
class eoEvalFuncCounterBounderException : public std::exception
{
public:
    eoEvalFuncCounterBounderException(unsigned long threshold) : _threshold(threshold){}

    const char* what() const throw()
    {
        std::ostringstream ss;
        ss << "STOP in eoEvalFuncCounterBounderException: the maximum number of evaluation has been reached (" << _threshold << ").";
        return ss.str().c_str();
    }

private:
    unsigned long _threshold;
};

/** Counts the number of evaluations actually performed and throw an eoEvalFuncCounterBounderException
 * when the maximum number of allowed evaluations is reached.
 *
 * This eval counter permits to stop a search during a generation, without waiting for a continue to be
 * checked at the end of the loop. Useful if you have 10 individuals and 10 generations,
 * but want to stop after 95 evaluations.
*/
template < typename EOT >
class eoEvalFuncCounterBounder : public eoEvalFuncCounter< EOT >
{
public :
    eoEvalFuncCounterBounder(eoEvalFunc<EOT>& func, unsigned long threshold, std::string name = "Eval. ")
        : eoEvalFuncCounter< EOT >( func, name ), _threshold( threshold )
    {}

    using eoEvalFuncCounter< EOT >::value;

    virtual void operator()(EOT& eo)
    {
        if (eo.invalid())
            {
                value()++;

                if (_threshold > 0 && value() >= _threshold)
                    {
                        throw eoEvalFuncCounterBounderException(_threshold);
                    }

                func(eo);
            }
    }

private :
    unsigned long _threshold;
};

/** @} */
#endif
