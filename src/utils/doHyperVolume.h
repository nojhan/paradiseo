// (c) Thales group, 2010
/*
    Authors:
             Johann Dreo <johann.dreo@thalesgroup.com>
             Caner Candan <caner.candan@thalesgroup.com>
*/

#ifndef _doHyperVolume_h
#define _doHyperVolume_h

template < typename EOT >
class doHyperVolume
{
public:
    typedef typename EOT::AtomType AtomType;

    doHyperVolume() : _hv(1) {}

    void update(AtomType v)
    {
	_hv *= ::sqrt( v );

	assert( _hv <= std::numeric_limits< AtomType >::max() );
    }

    AtomType get_hypervolume() const { return _hv; }

protected:
    AtomType _hv;
};

#endif // !_doHyperVolume_h
