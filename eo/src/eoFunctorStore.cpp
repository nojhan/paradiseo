#include <eoFunctorStore.h>
#include <eoFunctor.h>

/// clears the memory
eoFunctorStore::~eoFunctorStore()
{
    for (size_t i = 0; i < vec.size(); ++i)
    {
        delete vec[i];
    }
}
