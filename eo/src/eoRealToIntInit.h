#ifndef eoRealToIntInit_h_INCLUDED
#define eoRealToIntInit_h_INCLUDED

#include "es/eoReal.h"
#include "utils/eoIntBounds.h"

template<class EOTINT, class EOTREAL = eoReal<typename EOTINT::FitnessType>>
class eoRealToIntInit : public eoInit<EOTINT>
{
public:

    using EOTreal = EOTREAL;

    enum Repair {
        folds,
        truncate
    };

    eoRealToIntInit( eoInit<EOTreal>& init ) :
        _whenout(Repair::truncate),
        _nobounds(),
        _bounds(_nobounds),
        _init(init)
    { }

    eoRealToIntInit( eoInit<EOTreal>& init, eoIntBounds& bounds, Repair whenout = Repair::truncate ) :
        _whenout(whenout),
        _nobounds(),
        _bounds(bounds),
        _init(init)
    { }

    virtual void operator()(EOTINT& intsol) override
    {
        #ifndef NDEBUG
            for(size_t i=0; i < intsol.size(); ++i) {
                assert(_bounds.isInBounds(intsol[i]));
            }
        #endif

        EOTreal floatsol;
        std::copy( std::begin(intsol), std::end(intsol), std::back_inserter(floatsol) );

        _init(floatsol);

        intsol.resize(floatsol.size());

        for(size_t i=0; i < floatsol.size(); ++i) {
            typename EOTreal::AtomType rounded = std::round(floatsol[i]);
            if( not _bounds.isInBounds(rounded) ) {
                switch(_whenout) {
                    case Repair::truncate:
                        _bounds.truncate(rounded);
                        break;
                    case Repair::folds:
                        _bounds.foldsInBounds(rounded);
                        break;
                }
            }
            intsol[i] = static_cast<typename EOTINT::AtomType>(rounded);
        }
    }

protected:
    Repair _whenout;
    eoIntNoBounds _nobounds;

    eoIntBounds& _bounds;
    eoInit<EOTreal>& _init;
};



#endif // eoRealToIntInit_h_INCLUDED
