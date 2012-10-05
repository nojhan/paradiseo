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

#ifndef SYMCOMPILE_H_
#define SYMCOMPILE_H_

#include <vector>
#include <sym/Sym.h>

typedef double (*single_function)(const double []);
typedef double (*multi_function)(const double[], double[]);

/* 
 * Important, after every call of the functions below, the function pointers of the previous
 * call are invalidated. Sorry, but that's the way the cookie crumbles (in tcc)
 * */

single_function compile(Sym sym);
multi_function  compile(const std::vector<Sym>& sym);
void compile(const std::vector<Sym>& sym, std::vector<single_function>& functions);

#endif
