//-----------------------------------------------------------------------------
// vecop.h
//-----------------------------------------------------------------------------

#ifndef VECOP_H
#define VECOP_H

//-----------------------------------------------------------------------------

#include <iostream>    // ostream istream
#include <vector>      // vector
#include <functional>  // plus minus multiplies divides
#include <numeric>     // inner_product

//-----------------------------------------------------------------------------
// vector + vector
//-----------------------------------------------------------------------------

template<class T> vector<T> operator+(const vector<T>& v1, const vector<T>& v2)
{
  vector<T> tmp = v1;
  transform(tmp.begin(), tmp.end(), v2.begin(), tmp.begin(), plus<T>());
  return tmp;
}

template<class T> vector<T> operator-(const vector<T>& v1, const vector<T>& v2)
{
  vector<T> tmp = v1;
  transform(tmp.begin(), tmp.end(), v2.begin(), tmp.begin(), minus<T>());
  return tmp;
}

template<class T> T operator*(const vector<T>& v1, const vector<T>& v2)
{
  return inner_product(v1.begin(), v1.end(), v2.begin(), static_cast<T>(0));
}

template<class T> T operator/(const vector<T>& v1, const vector<T>& v2)
{
  return inner_product(v1.begin(), v1.end(), v2.begin(), static_cast<T>(0),
		       plus<T>(), divides<T>());
}

//-----------------------------------------------------------------------------
// vector += vector
//-----------------------------------------------------------------------------

template<class T> vector<T>& operator+=(vector<T>& v1, const vector<T>& v2)
{
  transform(v1.begin(), v1.end(), v2.begin(), v1.begin(), plus<T>());
  return v1;
}

template<class T> vector<T>& operator-=(vector<T>& v1, const vector<T>& v2)
{
  transform(v1.begin(), v1.end(), v2.begin(), v1.begin(), minus<T>());
  return v1;
}

//-----------------------------------------------------------------------------
// vector + number
//-----------------------------------------------------------------------------

template<class A, class B> vector<A> operator+(const vector<A>& a, const B& b)
{
  vector<A> tmp = a;
  transform(tmp.begin(), tmp.end(), tmp.begin(), bind2nd(plus<A>(), b));
  return tmp;
}

template<class A, class B> vector<A> operator-(const vector<A>& a, const B& b)
{
  vector<A> tmp = a;
  transform(tmp.begin(), tmp.end(), tmp.begin(), bind2nd(minus<A>(), b));
  return tmp;
}

template<class A, class B> vector<A> operator*(const vector<A>& a, const B& b)
{
  vector<A> tmp = a;
  transform(tmp.begin(), tmp.end(), tmp.begin(), bind2nd(multiplies<A>(), b));
  return tmp;
}

template<class A, class B> vector<A> operator/(const vector<A>& a, const B& b)
{
  vector<A> tmp = a;
  transform(tmp.begin(), tmp.end(), tmp.begin(), bind2nd(divides<A>(), b));
  return tmp;
}

//-----------------------------------------------------------------------------
// number + vector
//-----------------------------------------------------------------------------

template<class A, class B> vector<A> operator+(const B& b, const vector<A>& a)
{
  vector<A> tmp = a;
  transform(tmp.begin(), tmp.end(), tmp.begin(), bind2nd(plus<A>(), b));
  return tmp;
}

template<class A, class B> vector<A> operator-(const B& b, const vector<A>& a)
{
  vector<A> tmp(a.size(), b);
  transform(tmp.begin(), tmp.end(), a.begin(), tmp.begin(), minus<A>());
  return tmp;
}

template<class A, class B> vector<A> operator*(const B& b, const vector<A>& a)
{
  vector<A> tmp = a;
  transform(tmp.begin(), tmp.end(), tmp.begin(), bind2nd(multiplies<A>(), b));
  return tmp;
}

template<class A, class B> vector<A> operator/(const B& b, const vector<A>& a)
{
  vector<A> tmp(a.size(), b);
  transform(tmp.begin(), tmp.end(), a.begin(), tmp.begin(), divides<A>());
  return tmp;
}

//-----------------------------------------------------------------------------
// vector += number
//-----------------------------------------------------------------------------

template<class A, class B> vector<A>& operator+=(vector<A>& a, const B& b)
{
  transform(a.begin(), a.end(), a.begin(), bind2nd(plus<A>(), b));
  return a;
}

template<class A, class B> vector<A>& operator-=(vector<A>& a, const B& b)
{
  transform(a.begin(), a.end(), a.begin(), bind2nd(minus<A>(), b));
  return a;
}

template<class A, class B> vector<A>& operator*=(vector<A>& a, const B& b)
{
  transform(a.begin(), a.end(), a.begin(), bind2nd(multiplies<A>(), b));
  return a;
}

template<class A, class B> vector<A>& operator/=(vector<A>& a, const B& b)
{
  transform(a.begin(), a.end(), a.begin(), bind2nd(divides<A>(), b));
  return a;
}

//-----------------------------------------------------------------------------
// I/O
//-----------------------------------------------------------------------------

template<class T> ostream& operator<<(ostream& os, const vector<T>& v)
{
  os << '<';
  if (v.size())
    {
      copy(v.begin(), v.end() - 1, ostream_iterator<T>(os, " "));
      os << v.back();
    } 
  return os << '>';
}

template<class T> istream& operator>>(istream& is, vector<T>& v)
{
  v.clear();
  
  char c;
  is >> c;
  if (!is || c != '<')
    is.setstate(ios::failbit);
  else
    {
      T t;
      do {
	is >> c;
	if (is && c!= '>')
	  {
	    is.putback(c);
	    is >> t;
	    if (is)
	      v.push_back(t);
	  }
      } while (is && c != '>');
    }
  
  return is;
}

//-----------------------------------------------------------------------------
// euclidean_distance
//-----------------------------------------------------------------------------

template<class T> T euclidean_distance(const vector<T>& v1, 
				       const vector<T>& v2)
{
  T sum = 0, tmp;
  
  for (unsigned i = 0; i < v1.size(); ++i)
    {
      tmp = v1[i] - v2[i];
      sum += tmp * tmp;
    }

  return sqrt(sum);
}

//-----------------------------------------------------------------------------

#endif

