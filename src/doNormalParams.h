#ifndef _doNormalParams_h
#define _doNormalParams_h

template < typename EOT >
class doNormalParams
{
public:
    doNormalParams(EOT mean, EOT variance)
	: _mean(mean), _variance(variance)
    {
	assert(_mean.size() > 0);
	assert(_mean.size() == _variance.size());
    }

    EOT& mean(){return _mean;}
    EOT& variance(){return _variance;}

    unsigned int size()
    {
	assert(_mean.size() == _variance.size());
	return _mean.size();
    }

private:
    EOT _mean;
    EOT _variance;
};

#endif // !_doNormalParams_h
