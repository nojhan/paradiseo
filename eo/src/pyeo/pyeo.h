/*
    pyeo
    
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

#ifndef PYEO_H
#define PYEO_H

#include <EO.h>
#include <boost/python.hpp>

using namespace boost::python;
/*
class PyFitness : public boost::python::object
{
    public :
    PyFitness() : object() {}
    
    template <class T>
    PyFitness(const T& o) : object(o) {}
   
    
    static unsigned nObjectives() { return 1; }
    static double tol() { return 1e-6; }
    static bool maximizing(int which) { return true; }

    static bool dominates(const PyFitness& one, const PyFitness& oth)  { return true; } // for the moment
    
    operator unsigned() const { return 1; } // for the moment
    operator double() const { return 1; } // for the moment
    
    PyFitness operator[](int i) { return PyFitness(object::operator[](i)); }
   
    friend ostream& operator<<(ostream& os, const PyFitness& p) { const object& o = p; os << o; return os; }
    friend istream& operator>>(istream& is, PyFitness& p) { object o; is >> o; p = o; return is; }
    
    typedef PyFitness AtomType;
};



object fabs(object obj) 
{
    object zero(0.0);
    if (obj < zero ) 
	return zero-obj; 
    return obj; 
}

object max(object a, object b)
{
    if (a < b)
	return b;
    return a;
}
*/

struct PyEO : public EO< object  >
{  
    typedef object Fitness;
    
    Fitness getFitness() const { return invalid()? Fitness(): fitness(); }
    void setFitness(Fitness f) { if (f == Fitness()) invalidate(); else fitness(f); }

    object getGenome() const { return genome; }
    void setGenome(object g) { genome = g; }
    object genome;

    std::string to_string() const
    {
	std::string result;
	result += extract<const char*>(str(getFitness()));
	result += ' ';
	result += extract<const char*>(str(genome));
	return result;
    }

    bool operator<(const PyEO& other) const { return EO<Fitness>::operator<(other); }
    bool operator>(const PyEO& other) const { return EO<Fitness>::operator>(other); }

};

ostream& operator<<(ostream& os, const PyEO& _eo);

struct PyEO_pickle_suite : boost::python::pickle_suite
{
    typedef PyEO::Fitness Fitness;
    
    static
    boost::python::tuple getstate(const PyEO& _eo)
    {
	return make_tuple(_eo.getFitness(), _eo.genome);
    }
    static
    void setstate(PyEO& _eo, boost::python::tuple pickled)
    {
	_eo.setFitness( Fitness(pickled[0]) );
	_eo.genome = pickled[1];
    }
};

#endif
