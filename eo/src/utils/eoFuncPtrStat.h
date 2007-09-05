#ifndef eoFuncPtrStat_h
#define eoFuncPtrStat_h

#include <eoFunctorStore.h>
#include <utils/eoStat.h>

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

template <class EOT, class T>
eoFuncPtrStat<EOT, T>& makeFuncPtrStat( T (*func)(const eoPop<EOT>&), eoFunctorStore& store, std::string description = "func") {
    return store.storeFunctor(
        new eoFuncPtrStat<EOT, T>( func, description)
        );
}

#endif

