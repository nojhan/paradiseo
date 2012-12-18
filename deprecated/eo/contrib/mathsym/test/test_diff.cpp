#include <Sym.h>
#include <FunDef.h>
#include <iostream>

using namespace std;

int main() {
    
    Sym v = SymConst(1.2);

    Sym g = exp(-sqr(v));

    cout << g << endl;
    cout << differentiate(g, v.token()) << endl;
    
}

