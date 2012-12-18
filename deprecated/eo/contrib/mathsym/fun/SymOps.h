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

#ifndef SYMOPS_H
#define SYMOPS_H

#include "sym/Sym.h"

extern Sym operator+(Sym a, Sym b);
extern Sym operator*(Sym a, Sym b);
extern Sym operator/(Sym a, Sym b);
extern Sym operator-(Sym a, Sym b);
extern Sym pow(Sym a, Sym b);
extern Sym ifltz(Sym a, Sym b, Sym c);
extern Sym operator-(Sym a);

#endif
