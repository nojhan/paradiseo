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
// std::vector + std::vector
//-----------------------------------------------------------------------------

template<class T> std::vector<T> operator+(const std::vector<T>& v1, const std::vector<T>& v2)
{
  std::vector<T> tmp = v1;
  std::transform(tmp.begin(), tmp.end(), v2.begin(), tmp.begin(), std::plus<T>());
  return tmp;
}

template<class T> std::vector<T> operator-(const std::vector<T>& v1, const std::vector<T>& v2)
{
  std::vector<T> tmp = v1;
  std::transform(tmp.begin(), tmp.end(), v2.begin(), tmp.begin(), std::minus<T>());
  return tmp;
}

template<class T> T operator*(const std::vector<T>& v1, const std::vector<T>& v2)
{
  return inner_product(v1.begin(), v1.end(), v2.begin(), static_cast<T>(0));
}

template<class T> T operator/(const std::vector<T>& v1, const std::vector<T>& v2)
{
  return inner_product(v1.begin(), v1.end(), v2.begin(), static_cast<T>(0),
		       std::plus<T>(), std::divides<T>());
}

//-----------------------------------------------------------------------------
// std::vector += std::vector
//-----------------------------------------------------------------------------

template<class T> std::vector<T>& operator+=(std::vector<T>& v1, const std::vector<T>& v2)
{
  std::transform(v1.begin(), v1.end(), v2.begin(), v1.begin(), std::plus<T>());
  return v1;
}

template<class T> std::vector<T>& operator-=(std::vector<T>& v1, const std::vector<T>& v2)
{
  std::transform(v1.begin(), v1.end(), v2.begin(), v1.begin(), std::minus<T>());
  return v1;
}

//-----------------------------------------------------------------------------
// std::vector + number
//-----------------------------------------------------------------------------

template<class A, class B> std::vector<A> operator+(const std::vector<A>& a, const B& b)
{
  std::vector<A> tmp = a;
  std::transform(tmp.begin(), tmp.end(), tmp.begin(), std::bind2nd(std::plus<A>(), b));
  return tmp;
}

template<class A, class B> std::vector<A> operator-(const std::vector<A>& a, const B& b)
{
  std::vector<A> tmp = a;
  std::transform(tmp.begin(), tmp.end(), tmp.begin(), std::bind2nd(std::minus<A>(), b));
  return tmp;
}

template<class A, class B> std::vector<A> operator*(const std::vector<A>& a, const B& b)
{
  std::vector<A> tmp = a;
  std::transform(tmp.begin(), tmp.end(), tmp.begin(), std::bind2nd(std::multiplies<A>(), b));
  return tmp;
}

template<class A, class B> std::vector<A> operator/(const std::vector<A>& a, const B& b)
{
  std::vector<A> tmp = a;
  std::transform(tmp.begin(), tmp.end(), tmp.begin(), std::bind2nd(std::divides<A>(), b));
  return tmp;
}

//-----------------------------------------------------------------------------
// number + std::vector
//-----------------------------------------------------------------------------

template<class A, class B> std::vector<A> operator+(const B& b, const std::vector<A>& a)
{
  std::vector<A> tmp = a;
  std::transform(tmp.begin(), tmp.end(), tmp.begin(), std::bind2nd(std::plus<A>(), b));
  return tmp;
}

template<class A, class B> std::vector<A> operator-(const B& b, const std::vector<A>& a)
{
  std::vector<A> tmp(a.size(), b);
  std::transform(tmp.begin(), tmp.end(), a.begin(), tmp.begin(), std::minus<A>());
  return tmp;
}

template<class A, class B> std::vector<A> operator*(const B& b, const std::vector<A>& a)
{
  std::vector<A> tmp = a;
  std::transform(tmp.begin(), tmp.end(), tmp.begin(), bind2nd(std::multiplies<A>(), b));
  return tmp;
}

template<class A, class B> std::vector<A> operator/(const B& b, const std::vector<A>& a)
{
  std::vector<A> tmp(a.size(), b);
  std::transform(tmp.begin(), tmp.end(), a.begin(), tmp.begin(), std::divides<A>());
  return tmp;
}

//-----------------------------------------------------------------------------
// std::vector += number
//-----------------------------------------------------------------------------

template<class A, class B> std::vector<A>& operator+=(std::vector<A>& a, const B& b)
{
  std::transform(a.begin(), a.end(), a.begin(), std::bind2nd(std::plus<A>(), b));
  return a;
}

template<class A, class B> std::vector<A>& operator-=(std::vector<A>& a, const B& b)
{
  std::transform(a.begin(), a.end(), a.begin(), std::bind2nd(std::minus<A>(), b));
  return a;
}

template<class A, class B> std::vector<A>& operator*=(std::vector<A>& a, const B& b)
{
  std::transform(a.begin(), a.end(), a.begin(), std::bind2nd(std::multiplies<A>(), b));
  return a;
}

template<class A, class B> std::vector<A>& operator/=(std::vector<A>& a, const B& b)
{
  std::transform(a.begin(), a.end(), a.begin(), std::bind2nd(std::divides<A>(), b));
  return a;
}

//-----------------------------------------------------------------------------
// I/O
//-----------------------------------------------------------------------------

template<class T> std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
  os << '<';
  if (v.size())
    {
      std::copy(v.begin(), v.end() - 1, std::ostream_iterator<T>(os, " "));
      os << v.back();
    }
  return os << '>';
}

template<class T> std::istream& operator>>(std::istream& is, std::vector<T>& v)
{
  v.clear();

  char c;
  is >> c;
  if (!is || c != '<')
    is.setstate(std::ios::failbit);
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

template<class T> T euclidean_distance(const std::vector<T>& v1,
				       const std::vector<T>& v2)
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
