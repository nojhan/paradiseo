/*
<MWModel.cpp>
Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2012

Alexandre Quemy

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software.  You can  ue,
modify and/ or redistribute the software under the terms of the CeCILL
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info".

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or
data to be ensured and,  more generally, to use and operate it in the
same conditions as regards security.
The fact that you are presently reading this means that you have had
knowledge of the CeCILL license and that you accept its terms.

ParadisEO WebSite : http://paradiseo.gforge.inria.fr
Contact: paradiseo-help@lists.gforge.inria.fr
*/

template<template <class> class EOAlgo, class EOT, class Policy>
template<class... Args>
paradiseo::smp::MWModel<EOAlgo,EOT,Policy>::MWModel(unsigned workersNb, Args&... args) :
    EOAlgo<EOT>(args...),
    scheduler(workersNb)
{ 
    assert(workersNb > 0); 
}

template<template <class> class EOAlgo, class EOT, class Policy>
template<class... Args>
paradiseo::smp::MWModel<EOAlgo,EOT,Policy>::MWModel(Args&... args) :
    MWModel(std::thread::hardware_concurrency(), args...)
{}

template<template <class> class EOAlgo, class EOT, class Policy>
paradiseo::smp::MWModel<EOAlgo,EOT,Policy>::~MWModel() 
{}

template<template <class> class EOAlgo, class EOT, class Policy>
void paradiseo::smp::MWModel<EOAlgo,EOT,Policy>::apply(eoUF<EOT&, void>& func, eoPop<EOT>& pop) 
{     
    scheduler(func, pop);
}

template<template <class> class EOAlgo, class EOT, class Policy>
void paradiseo::smp::MWModel<EOAlgo,EOT,Policy>::evaluate(eoPop<EOT>& pop) 
{     
    scheduler(this->eval, pop);
}

template<template <class> class EOAlgo, class EOT, class Policy>
void paradiseo::smp::MWModel<EOAlgo,EOT,Policy>::operator()(eoPop<EOT>& pop) 
{
    // Call the tag dispatcher
    operator()(pop,typename traits<EOAlgo,EOT>::type());
}

template<template <class> class EOAlgo, class EOT, class Policy>
void paradiseo::smp::MWModel<EOAlgo,EOT,Policy>::operator()(eoPop<EOT>& pop, const error_tag&) 
{
    throw std::runtime_error("This is not a valid algorithm");
}

