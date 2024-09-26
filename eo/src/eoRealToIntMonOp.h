#ifndef eoRealToIntMonOp_h_INCLUDED
#define eoRealToIntMonOp_h_INCLUDED

#include "es/eoReal.h"
#include "utils/eoIntBounds.h"

template<class EOTINT, class EOTREAL = eoReal<typename EOTINT::FitnessType>>
class eoRealToIntMonOp : public eoMonOp<EOTINT>
{
public:

    using EOTreal = EOTREAL;

    enum Repair {
        folds,
        truncate
    };

    eoRealToIntMonOp( eoMonOp<EOTreal>& monop ) :
        _whenout(Repair::truncate),
        _nobounds(),
        _bounds(_nobounds),
        _monop(monop)
    { }

    eoRealToIntMonOp( eoMonOp<EOTreal>& monop, eoIntBounds& bounds, Repair whenout = Repair::truncate ) :
        _whenout(whenout),
        _nobounds(),
        _bounds(bounds),
        _monop(monop)
    { }

    bool operator()(EOTINT& intsol)
    {
        #ifndef NDEBUG
            for(size_t i=0; i < intsol.size(); ++i) {
                assert(_bounds.isInBounds(intsol[i]));
            }
        #endif

        EOTreal floatsol;
        std::copy( std::begin(intsol), std::end(intsol), std::back_inserter(floatsol) );

        bool changed = _monop(floatsol);

        if(changed) {
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
        return changed;
    }

protected:
    Repair _whenout;
    eoIntNoBounds _nobounds;

    eoIntBounds& _bounds;
    eoMonOp<EOTreal>& _monop;
};

#endif // eoRealToIntMonOp_h_INCLUDED
