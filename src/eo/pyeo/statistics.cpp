#include "../utils/eoStat.h"

#include "PyEO.h"
#include "valueParam.h"

using namespace boost::python;

class StatBaseWrapper : public eoStatBase<PyEO>
{
public:
    PyObject* self;
    StatBaseWrapper(PyObject* p) : self(p) {}

    void operator()(const eoPop<PyEO>& pop)
    {
	call_method<void>(self, "__call__", boost::ref(pop));
    }
};

class SortedStatBaseWrapper : public eoSortedStatBase<PyEO>
{
public:
    PyObject* self;
    SortedStatBaseWrapper(PyObject* p) : self(p) {}

    void operator()(const std::vector<const PyEO*>& pop)
    {
	call_method<void>(self, "__call__", boost::ref(pop));
    }
};

typedef std::vector<const PyEO*> eoPopView;

const PyEO& popview_getitem(const std::vector<const PyEO*>& pop, int it)
{
    unsigned item = unsigned(it);
    if (item > pop.size())
	throw index_error("too much");

    return *pop[item];
}

void statistics()
{
    class_<eoStatBase<PyEO>, StatBaseWrapper, boost::noncopyable>
	("eoStatBase", init<>())
	.def("lastCall", &eoStatBase<PyEO>::lastCall)
	.def("__call__", &StatBaseWrapper::operator())
	;

    class_< eoPopView >("eoPopView")
	.def("__getitem__", popview_getitem, return_internal_reference<>() )
	.def("__len__", &eoPopView::size)
	;

    class_<eoSortedStatBase<PyEO>, SortedStatBaseWrapper, boost::noncopyable>
	("eoSortedStatBase", init<>())
	.def("lastCall", &eoSortedStatBase<PyEO>::lastCall)
	.def("__call__", &SortedStatBaseWrapper::operator())
	;
}
