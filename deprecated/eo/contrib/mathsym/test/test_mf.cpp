
#include "Sym.h"
#include "MultiFunction.h"
#include "FunDef.h"

using namespace std;

int main() {

    Sym v = SymVar(0);
    Sym c = SymConst(0.1);

    Sym sym = inv(v) + c;
    Sym a = sym;
    
    sym = sym * sym;
    Sym b = sym;
    sym = sym + sym;
    
    c = sym;
    
    vector<Sym> pop;
    pop.push_back(sym);

    MultiFunction m(pop);


    vector<double> vec(1);
    vec[0] = 10.0;
    cout << sym << endl;

    cout << "Eval " << eval(sym, vec);
    
    vector<double> y(1);

    m(vec,y);

    cout << " " << y[0] << endl;
    
    cout << "3 " << eval(a,vec) << endl;
    cout << "4 " << eval(b, vec) << endl;
    cout << "5 " << eval(c, vec) << endl;
    
}

