#ifndef _doDistribParams_h
#define _doDistribParams_h

#include <vector>

template < typename EOT >
class doDistribParams
{
public:
    doDistribParams(unsigned n = 2)
	: _params(n)
    {}

    doDistribParams(const doDistribParams& p) { *this = p; }

    doDistribParams& operator=(const doDistribParams& p)
    {
	if (this != &p)
	    {
		this->_params = p._params;
	    }

	return *this;
    }

    EOT& param(unsigned int i = 0){return _params[i];}

    unsigned int param_size(){return _params.size();}

    unsigned int size()
    {
	for (unsigned int i = 0, size = param_size(); i < size - 1; ++i)
	    {
		assert(param(i).size() == param(i + 1).size());
	    }

	return param(0).size();
    }

private:
    std::vector< EOT > _params;
};

#endif // !_doDistribParams_h
