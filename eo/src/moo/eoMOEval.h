#ifndef eoMOEval_h
#define eoMOEval_h

#include <eoPopEvalFunc.h>

template <class EOT>
class eoMOEval : public eoPopEvalFunc<EOT> {
   
    protected: 
    eoMOEval(eoEvalFunc<EOT>& eval)  : default_loop(eval), pop_eval(default_loop) {}
    eoMOEval(eoPopEvalFunc<EOT>& ev) : default_loop(dummy_eval), pop_eval(ev) {}
  
    void eval(eoPop<EOT>& parents, eoPop<EOT>& offspring) {
        pop_eval(parents, offspring);
    }
    
    private :

    class eoDummyEval : public eoEvalFunc<EOT> {public: void operator()(EOT &) {} } dummy_eval;
    eoPopLoopEval<EOT> default_loop;
    eoPopEvalFunc<EOT>& pop_eval;

};


#endif
