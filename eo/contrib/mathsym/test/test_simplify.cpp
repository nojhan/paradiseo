
#include <FunDef.h>

using namespace std;

int main() {
    
    Sym c1 = SymConst(0.4);
    Sym c2 = SymConst(0.3);
    Sym v1 = SymVar(0);

    Sym expr = (c1 + c2) * ( (c1 + c2) * v1);
    
    cout << expr << endl;
    cout << simplify(expr) << endl;
    
}

