//-----------------------------------------------------------------------------
// eoNonUniform.h
//-----------------------------------------------------------------------------

#ifndef EONONUNIFORM_H
#define EONONUNIFORM_H

//-----------------------------------------------------------------------------

#include <math.h>  // pow

//-----------------------------------------------------------------------------
// eoNonUniform
//-----------------------------------------------------------------------------

class eoNonUniform
{
public:
  eoNonUniform(const unsigned _num_step):
    step_value(0), num_step_value(_num_step) {}
  
  void reset() { step_value = 0; }
  
  const unsigned& step() const { return step_value; }
  const unsigned& num_step() const { return num_step_value; }
  
  operator int() const { return step_value < num_step_value; }
  
  void operator++() { ++step_value; }
  void operator++(int) { ++step_value; }
  
private:
  unsigned step_value, num_step_value;
};

//-----------------------------------------------------------------------------
// eoLinear
//-----------------------------------------------------------------------------

class eoLinear
{
public:
  eoLinear(const double        _first, 
	   const double        _last, 
	   const eoNonUniform& _non_uniform):
    first(_first), 
    diff((_last - _first) / (_non_uniform.num_step() - 1)), 
    non_uniform(_non_uniform) {}
  
  double operator()() const
  {
    return first + diff * non_uniform.step();
  }
  
private:
  double              first, diff;
  const eoNonUniform& non_uniform;
};

//-----------------------------------------------------------------------------
// eoNegExp2
//-----------------------------------------------------------------------------

class eoNegExp2
{
 public:
  eoNegExp2(const double        _r,
	    const double        _b,
	    const eoNonUniform& _non_uniform):
    r(_r), b(_b), 
    non_uniform(_non_uniform) {}
  
  double operator()() const
    {
      return 1.0 - pow(r, pow(1.0 - (double)non_uniform.step() / 
			      non_uniform.num_step(), b));
    }
  
 private:
  double              r, b;
  const eoNonUniform& non_uniform;
};

//-----------------------------------------------------------------------------

#endif NON_UNIFORM_HH
