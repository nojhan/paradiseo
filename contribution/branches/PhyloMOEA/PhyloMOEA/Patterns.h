/***************************************************************************
 *   Copyright (C) 2005 by Waldo Cancino                                   *
 *   wcancino@icmc.usp.br                                                  *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#ifndef PATTERNS_H
#define PATTERNS_H
#include <Sequences.h>
#include <vector>
#include <string>

class Patterns
{
	private:
		int num_patterns;
		
		double freqs[4];
		std::vector<unsigned char*> pattern;
		std::vector<int> pattern_count;
		Sequences &seqs;
	public:
		// return a pattern content
		Patterns(Sequences &S);
		~Patterns();
		unsigned char* operator[](int n) const { return pattern[n]; }
		inline int count(int n) const { return pattern_count[n]; }
		inline int count() const { return num_patterns; }
		inline double *frequences() { return freqs; }
		inline double frequence(int i) const { return freqs[i]; }
		inline std::string pattern_name(int i) const { return seqs.seqname(i); }
		inline int search_taxon_name(std::string &s) const { return seqs.search_taxon_name(s); }
		inline bool is_ambiguous(short int l) const { return seqs.is_ambiguous(l); }
		inline bool is_gap(short int l) const { return seqs.is_gap(l); }
		inline bool is_undefined(short int l) const { return seqs.is_undefined(l); }
		inline unsigned char* ambiguos_meaning(short int l) const { return seqs.ambiguos_meaning(l); }
		void calculate_frequences();
		int num_sequences();
};
#endif
