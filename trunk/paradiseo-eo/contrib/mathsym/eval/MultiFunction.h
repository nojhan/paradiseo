#ifndef MULTIFUNCTION_H_
#define MULTIFUNCTION_H_

#include <vector>

class Sym;
class MultiFunctionImpl;

class MultiFunction {
    MultiFunction& operator=(const MultiFunction&);
    MultiFunction(const MultiFunction&);

    MultiFunctionImpl* pimpl;
    
    public:

    MultiFunction(const std::vector<Sym>& pop);
    ~MultiFunction();
    
    void operator()(const std::vector<double>& x, std::vector<double>& y);
    void operator()(const double* x, double* y);
    
};

#endif

