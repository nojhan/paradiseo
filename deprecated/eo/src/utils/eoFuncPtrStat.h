#ifndef eoFuncPtrStat_h
#define eoFuncPtrStat_h

#include <eoFunctorStore.h>
#include <utils/eoStat.h>



/** Wrapper to turn any stand-alone function and into an eoStat.
 *
 * The function should take an eoPop as argument.
 *
 * @ingroup Stats
 */
template <class EOT, class T>
class eoFuncPtrStat : public eoStat<EOT, T>
{
public :
    typedef T (*func_t)(const eoPop<EOT>&);


    eoFuncPtrStat(func_t f, std::string _description = "func_ptr")
        : eoStat<EOT, T>(T(), _description), func(f)
        {}

    using eoStat<EOT, T>::value;

    void operator()(const eoPop<EOT>& pop) {
        value() = func(pop);
    }

private:
    func_t func;
};

/**
 * @ingroup Stats
 */
template <class EOT, class T>
eoFuncPtrStat<EOT, T>& makeFuncPtrStat( T (*func)(const eoPop<EOT>&), eoFunctorStore& store, std::string description = "func") {
    return store.storeFunctor(
        new eoFuncPtrStat<EOT, T>( func, description)
        );
}

/** Wrapper to turn any stand-alone function and into an eoStat.
 *
 * The function should take an eoPop as argument.
 *
 * @ingroup Stats
 */
template <class EOT, class T>
class eoFunctorStat : public eoStat<EOT, T>
{
public :
    eoFunctorStat(eoUF< const eoPop<EOT>&, T >& f, std::string _description = "functor")
        : eoStat<EOT, T>(T(), _description), func(f)
        {}

    using eoStat<EOT, T>::value;

    void operator()(const eoPop<EOT>& pop) {
        value() = func(pop);
    }

private:
    eoUF< const eoPop<EOT>&, T >& func;
};

/**
 * @ingroup Stats
 */
template <class EOT, class T>
eoFunctorStat<EOT, T>& makeFunctorStat( eoUF< const eoPop<EOT>&, T >& func, eoFunctorStore& store, std::string description = "func") {
    return store.storeFunctor(
        new eoFunctorStat<EOT, T>( func, description)
        );
}

#endif
