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

#ifndef LANGUAGE_TABLE_H
#define LANGUAGE_TABLE_H

#include <sym/token.h>

class LanguageImpl;
class Sym;

class LanguageTable {
    
    LanguageImpl* pimpl;

    public:
    
    LanguageTable();
    ~LanguageTable();

    LanguageTable(const LanguageTable& org);

    LanguageTable& operator=(const LanguageTable& org);
    
    /* setting it up */
    typedef Sym (*erc_func)();
    
    void add_function(token_t token, unsigned arity);
    void add_function(functor_t functor); 
    void set_erc(erc_func func);
    
    /* Getting info out */
    
    Sym get_random_var() const;
    Sym get_random_const() const;
    
    functor_t get_random_function() const;
    token_t  get_random_function(token_t org, unsigned arity) const;
};

#endif

