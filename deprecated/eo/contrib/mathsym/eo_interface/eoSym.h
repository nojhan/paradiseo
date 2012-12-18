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

#ifndef EOSYM_H_
#define EOSYM_H_

#include <EO.h>
#include <sym/Sym.h>
#include <fun/FunDef.h>


template <class Fitness>
class EoSym : public EO<Fitness>, public Sym {

    public:
	
	void set(const Sym& sym) {
	    EO<Fitness>::invalidate();
	    static_cast<Sym*>(this)->operator=(sym);
	}

	Sym& get()      { return static_cast<Sym&>(*this); };
	Sym get() const { return static_cast<Sym&>(*this); };
	
	virtual void printOn(std::ostream& os) const;
	virtual void readFrom(std::istream& is);
};


template <class Fitness>
void EoSym<Fitness>::printOn(std::ostream& os) const {
    EO<Fitness>::printOn(os);
    os << ' ';
    write_raw(os, *this);
}

template <class Fitness>
void EoSym<Fitness>::readFrom(std::istream& is) {
    EO<Fitness>::readFrom(is);
    read_raw(is, *this); 
}

template <class Fitness>
inline std::ostream& operator<<(std::ostream& os, const EoSym<Fitness>& f) { f.printOn(os); return os; }
template <class Fitness>
inline std::istream& operator>>(std::istream& is, EoSym<Fitness>& f) { f.readFrom(is); return is; }


#endif


