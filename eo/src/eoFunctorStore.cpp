#ifdef _MSC_VER
// to avoid long name warnings
#pragma warning(disable:4786)
#endif

#include <cstddef>

#include <paradiseo/eo/eoFunctorStore.h>
#include <paradiseo/eo/eoFunctor.h>


/// clears the memory
eoFunctorStore::~eoFunctorStore()
{
    for( std::size_t i = 0; i < vec.size(); ++i) {
        delete vec[i];
    }
}
