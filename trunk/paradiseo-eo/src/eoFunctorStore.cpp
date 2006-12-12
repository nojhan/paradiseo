#ifdef _MSC_VER
// to avoid long name warnings
#pragma warning(disable:4786)
#endif

#include <eoFunctorStore.h>
#include <eoFunctor.h>


/// clears the memory
eoFunctorStore::~eoFunctorStore()
{
    for(size_t i = 0; i < vec.size(); ++i) {
        delete vec[i];
    }
}
