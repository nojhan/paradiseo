//-----------------------------------------------------------------------------
// eoNonUniform.h
//-----------------------------------------------------------------------------

#ifndef eoNonUniform_h
#define eoNonUniform_h

//-----------------------------------------------------------------------------
// eoNonUniform: base class for non uniform operators
//-----------------------------------------------------------------------------

template<class Time> class eoNonUniform
{
public:
  eoNonUniform(const Time& _time = Time(), const Time& _max_time = Time()): 
    time_value(_time), max_time_value(_max_time) {}
  
  const Time& time() const { return time_value; }
  const Time& max_time() const { return max_time_value; }
  
private:
  Time &time_value, &max_time_value;
};

//-----------------------------------------------------------------------------

#endif eoNonUniform_h
