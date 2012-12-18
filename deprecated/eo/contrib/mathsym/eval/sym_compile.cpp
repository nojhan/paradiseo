/*	    
 *             Copyright (C) 2005 Maarten Keijzer
 *
 *          This program is free software; you can redistribute it and/or modify
 *          it under the terms of version 2 of the GNU General Public License as 
 *          published by the Free Software Foundation. 
 *
 *          This program is distributed in the hope that it will be useful,
 *          but WITHOUT ANY WARRANTY; without even the implied warranty of
 *          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *          GNU General Public License for more details.
 *
 *          You should have received a copy of the GNU General Public License
 *          along with this program; if not, write to the Free Software
 *          Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */


#include "Sym.h"
#include "FunDef.h"
#include "sym_compile.h"

#include <sstream>

using namespace std;

extern "C" {
    void  symc_init();
    int  symc_compile(const char* func_str);
    int  symc_link();
    void* symc_get_fun(const char* func_name);
    void* symc_make(const char* func_str, const char* func_name);
}

string make_prototypes() {
    string prot = get_prototypes();
    prot += "double sqr(double x) { return x*x; }";
    return prot;
}

// contains variable names, like 'a0', 'a1', etc. or regular code

#if USE_TR1 
typedef std::tr1::unordered_map<Sym, string, HashSym> HashMap;
#else
typedef hash_map<Sym, string, HashSym> HashMap;
#endif

// prints 'num' in reverse notation. Does not matter as it's a unique id
string make_var(unsigned num) {
    string str = "a";
    do {
	str += char('0' + (num % 10));
	num /= 10;
    } while (num);
    return str;
}

template <class T>
string to_string(T t) {
    ostringstream os;
    os << t;
    return os.str();
}


HashMap::iterator find_entry(const Sym& sym, string& str, HashMap& map) {
    HashMap::iterator result = map.find(sym);

    if (result == map.end()) { // new entry
	const SymVec& args = sym.args();
	
	vector<string> argstr(args.size());
	for (unsigned i = 0; i < args.size(); ++i) {
	    argstr[i] = find_entry(args[i], str, map)->second;
	}

	string var = make_var(map.size()); // map.size(): unique id
	string code;	
	// write out the code
	const FunDef& fun = get_element(sym.token());
	code = fun.c_print(argstr, vector<string>() );
	    
	str += "double " + var + "=" + code + ";\n";
	result = map.insert( make_pair(sym, var ) ).first; // only want iterator
    }
    
    return result;
}

void write_entry(const Sym& sym, string& str, HashMap& map, unsigned out) {
    HashMap::iterator it = find_entry(sym, str, map);
    
    str += "y[" + to_string(out) + "]=" + it->second + ";\n";
    //cout << "wrote " << out << '\n';
}

#include <fstream>
multi_function compile(const std::vector<Sym>& syms) {
    
    //cout << "Multifunction " << syms.size() << endl;
    // static stream to avoid fragmentation of these LARGE strings
    static string str;
    str.clear();
    str += make_prototypes();

    str += "extern double func(const double* x, double* y) { \n ";
   
    multi_function result;
    HashMap map(Sym::get_dag().size());
    
    for (unsigned i = 0; i < syms.size(); ++i) {
	write_entry(syms[i], str, map, i);
    }
    
    str += ";}";

    
    /*static int counter = 0;
    ostringstream nm;
    nm << "cmp/compiled" << (counter++) << ".c";
    cout << "Saving as " << nm.str() << endl;
    ofstream cmp(nm.str().c_str());
    cmp << str;
    cmp.close();

    //cout << "Multifunction " << syms.size() << endl;
    cout << "Size of map " << map.size() << endl;
*/

    result = (multi_function) symc_make(str.c_str(), "func"); 

    if (result==0) { // error
	cout << "Error in compile " << endl;
    }

    return result;
}

single_function compile(Sym sym) {

    ostringstream os;

    os << make_prototypes();
    os << "double func(const double* x) { return ";
    
    string code = c_print(sym);
    os << code;
    os << ";}";
    string func_str = os.str();
  
    //cout << "compiling " << func_str << endl;
    
    return  (single_function) symc_make(func_str.c_str(), "func"); 
}

/* finds and inserts the full code in a hashmap */
HashMap::iterator find_code(Sym sym, HashMap& map) {
    HashMap::iterator result = map.find(sym);

    if (result == map.end()) { // new entry
	const SymVec& args = sym.args();
	vector<string> argstr(args.size());
	for (unsigned i = 0; i < args.size(); ++i) {
	    argstr[i] = find_code(args[i], map)->second;
	}

	// write out the code
	const FunDef& fun = get_element(sym.token());
	string code = fun.c_print(argstr, vector<string>());
	result = map.insert( make_pair(sym, code) ).first; // only want iterator
    }
    
    return result;
}

string print_code(Sym sym, HashMap& map) {
    HashMap::iterator it = find_code(sym, map);
    return it->second;
}

void compile(const std::vector<Sym>& syms, std::vector<single_function>& functions) {
    symc_init();
    
    static ostringstream os;
    os.str("");
    
    os << make_prototypes();
    HashMap map(Sym::get_dag().size());
    for (unsigned i = 0; i < syms.size(); ++i) {
	
	os << "double func" << i << "(const double* x) { return ";
	os << print_code(syms[i], map); //c_print(syms[i]);
	os << ";}\n";

	//symc_compile(os.str().c_str());
	//cout << "compiling " << os.str() << endl;	
    }

    os << ends;
#ifdef INTERVAL_DEBUG
    //cout << "Compiling " << os.str() << endl;
#endif
    
    symc_compile(os.str().c_str()); 
    symc_link();

    functions.resize(syms.size());
    for (unsigned i = 0; i < syms.size(); ++i) {
	ostringstream os2;
	os2 << "func" << i;
	
	functions[i] = (single_function) symc_get_fun(os2.str().c_str());
    }

}



