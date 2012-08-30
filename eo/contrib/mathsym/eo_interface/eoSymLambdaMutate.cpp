#include <eoSymLambdaMutate.h>
#include "FunDef.h"
#include "NodeSelector.h"

Sym compress(Sym sym, NodeSelector& sel) {
    
    return ::compress(sym);
    
    NodeSelector::NodeSelection s = sel.select_node(sym);
    
    Sym f = SymLambda( s.subtree());
    
    if (f == s.subtree()) { return sym; }
    
    return insert_subtree(sym, s.idx(), f);
}

extern Sym expand(Sym sym, NodeSelector& sel) {

    return ::expand_all(sym);
    
    NodeSelector::NodeSelection s = sel.select_node(sym);
    
    Sym f = SymUnlambda( s.subtree());
    
    if (f == s.subtree()) { return sym; }
    
    return insert_subtree(sym, s.idx(), f);
}
