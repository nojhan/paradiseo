#include <FunDef.h>

using namespace std;

int main() {

    Sym x = SymVar(0);
    Sym y = SymVar(1);

    Sym f = y + x*x;

    Sym l = SymLambda(f);
    
    SymVec args = l.args();
    args[0] = x;
    args[1] = y;
    l = Sym(l.token(), args);
    
    vector<double> v(3);
    v[0] = 2.0;
    v[1] = 3.0;
    v[2] = 4.0;

    double v1 = eval(f,v);
    double v2 = eval(l,v);

    cout << v1 << ' ' << v2 << endl;
    cout << f << endl;
    cout << l << endl;

    if (v1 != 7.0) return 1;
    if (v2 != 11.0) return 1;
}

