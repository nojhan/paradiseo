/*
    PyEO
    
    Copyright (C) 2003 Maarten Keijzer

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include <utils/eoRNG.h>
#include <boost/python.hpp>

#include <strstream>

using namespace boost::python;

namespace eo
{
    eoRng rng(time(0));
}

eoRng& get_rng() { return rng; }
double normal(eoRng& rng) { return rng.normal(); }

std::string rng_to_string(const eoRng& _rng)
{
    std::ostrstream os;
    _rng.printOn(os);
    os << std::ends;
    return os.str();
}

void rng_from_string(eoRng& _rng, std::string s)
{
    std::istrstream is(s.c_str(), s.size());
    _rng.readFrom(is);
}

void random_numbers()
{
    class_<eoRng, boost::noncopyable>("eoRng", init<uint32>())
	.def("flip", &eoRng::flip)
	.def("random", &eoRng::random)
	.def("rand", &eoRng::rand)
	.def("rand_max", &eoRng::rand_max)
	.def("reseed", &eoRng::reseed)
	.def("uniform", &eoRng::uniform)
	.def("normal", normal)
	.def("negexp", &eoRng::negexp)
	.def("to_string", rng_to_string)
	.def("from_string", rng_from_string)
	;
    
    def("rng", get_rng, return_value_policy<reference_existing_object>());
}
