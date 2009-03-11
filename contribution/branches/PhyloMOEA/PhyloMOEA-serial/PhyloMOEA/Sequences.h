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

#ifndef SEQUENCES_H
#define SEQUENCES_H
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cstring>

// class to store the sequences in a file

struct PatternInfo
{
	
	int idx;
	int count;
};

class Sequences
{
	private:
		//enum nucleotides = { A, C, G, T};
		unsigned int num_patterns;
		unsigned int num_inf_sites;
		unsigned int numseqs;
		unsigned int seqlen;
		unsigned char *seqs;
		std::vector<std::string> seqnames;
		std::vector<struct PatternInfo> patterns;
		std::vector<unsigned int> inf_sites;
		unsigned char meaning[12][5];
		double freqs[4];
		bool is_informative(unsigned int col) const;
	public:
		inline unsigned char* operator[](int n) const { return &(seqs[n*seqlen]); }
		Sequences(const char *filename);
		~Sequences() { 
			delete [] seqs;
		 }
		inline unsigned char seq_pos( int i, int j) const  { return (*this)[i][j]; }
		inline unsigned char pattern_pos(int i, int j) const 
			{ return seq_pos( j, patterns[i].idx); }
		inline unsigned char infsite_pos(int i, int j) const 
			{ return pattern_pos( inf_sites[i], j); }
		std::string position(int i);
		inline int num_seqs() const { return numseqs; }
		inline int seq_len() const { return seqlen; }
		inline int pattern_count() const { return num_patterns; }
		inline int pattern_count(int i) const { return patterns[i].count; }
		inline int infsite_count() const { return num_inf_sites; }
		inline int infsite_count(int i) const { return patterns[inf_sites[i]].count; }
		inline double *frequences() { return freqs; }
		inline double frequence(int i) const { return freqs[i]; }
		inline std::string seqname(int i) const { return seqnames.at(i); }
		inline bool is_defined(short int l) const { return l<4; }
		inline bool is_ambiguous(short int l) const { return (l>3 && l<14); }
		inline bool is_gap(short int l) const { return (l == 14); }
		inline bool is_undefined(short int l) const  { return (l ==15); }
		inline std::vector<struct PatternInfo>& get_patterns() { return patterns; }
		inline unsigned char* ambiguos_meaning(short int l) const { return 
			(unsigned char*) meaning[l-4]; }
		void list_taxons() const;
		int search_taxon_name(std::string &s) const;
		void calculate_patterns();
		void calculate_frequences();
		void save_seq_data(std::ostream &of);
		
};
#endif
	
