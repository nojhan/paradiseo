#ifndef eoRealToIntQuadOp_h_INCLUDED
#define eoRealToIntQuadOp_h_INCLUDED

#include "es/eoReal.h"
#include "utils/eoIntBounds.h"

template<class EOTINT, class EOTREAL = eoReal<typename EOTINT::FitnessType>>
class eoRealToIntQuadOp : public eoQuadOp<EOTINT>
{
public:

    using EOTreal = EOTREAL;

    enum Repair {
        folds,
        truncate
    };

    eoRealToIntQuadOp( eoQuadOp<EOTreal>& quadop ) :
        _whenout(Repair::truncate),
        _nobounds(),
        _bounds(_nobounds),
        _quadop(quadop)
    { }

    eoRealToIntQuadOp( eoQuadOp<EOTreal>& quadop, eoIntBounds& bounds, Repair whenout = Repair::truncate ) :
        _whenout(whenout),
        _nobounds(),
        _bounds(bounds),
        _quadop(quadop)
    { }

    bool operator()(EOTINT& intsol1, EOTINT& intsol2)
    {
        #ifndef NDEBUG
            for(size_t i=0; i < intsol1.size(); ++i) {
                assert(_bounds.isInBounds(intsol1[i]));
            }
            for(size_t i=0; i < intsol2.size(); ++i) {
                assert(_bounds.isInBounds(intsol2[i]));
            }
        #endif

        EOTreal floatsol1;
        std::copy( std::begin(intsol1), std::end(intsol1), std::back_inserter(floatsol1) );

        EOTreal floatsol2;
        std::copy( std::begin(intsol2), std::end(intsol2), std::back_inserter(floatsol2) );

        bool changed = _quadop(floatsol1, floatsol2);

        if(changed) {
            for(size_t i=0; i < floatsol1.size(); ++i) {
                typename EOTreal::AtomType rounded = std::round(floatsol1[i]);
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
                intsol1[i] = static_cast<typename EOTINT::AtomType>(rounded);
            }
            for(size_t i=0; i < floatsol2.size(); ++i) {
                typename EOTreal::AtomType rounded = std::round(floatsol2[i]);
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
                intsol2[i] = static_cast<typename EOTINT::AtomType>(rounded);
            }
        }
        return changed;
    }

protected:
    Repair _whenout;
    eoIntNoBounds _nobounds;

    eoIntBounds& _bounds;
    eoQuadOp<EOTreal>& _quadop;
};


#endif // eoRealToIntQuadOp_h_INCLUDED
