#include <eoFunctorStore.h>
#include <eoFunctor.h>

/// clears the memory
eoFunctorStore::~eoFunctorStore()
{
    for (int i = 0; i < vec.size(); ++i)
    {
        delete vec[i];
    }
}
