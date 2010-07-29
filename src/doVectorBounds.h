#ifndef _doVectorBounds_h
#define _doVectorBounds_h

template < typename EOT >
class doVectorBounds
{
public:
    doVectorBounds(EOT min, EOT max)
	: _min(min), _max(max)
    {
	assert(_min.size() > 0);
	assert(_min.size() == _max.size());
    }

    EOT& min(){return _min;}
    EOT& max(){return _max;}


    unsigned int size()
    {
	assert(_min.size() == _max.size());
	return _min.size();
    }

private:
    EOT _min;
    EOT _max;
};

#endif // !_doVectorBounds_h
