#ifndef eoEvalFuncCounterBounder_H
#define eoEvalFuncCounterBounder_H

#include <eoEvalFunc.h>
#include <utils/eoParam.h>

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

/**
Counts the number of evaluations actually performed, thus checks first
if it has to evaluate.. etc.
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

#endif
