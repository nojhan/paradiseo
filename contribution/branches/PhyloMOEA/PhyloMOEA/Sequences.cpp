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

#include <Sequences.h>
#include <iostream>
#include <fstream>
#include <map>
#include "assert.h"
#include <ExceptionManager.h>




void Sequences::list_taxons() const {
	std::cout << "lista de taxons" << std::endl;
	for(int i=0; i< seqnames.size(); i++)
	std::cout << seqnames[i] << std::endl;
}

using namespace std;

Sequences::Sequences(const char *fn)
{
	numseqs = 0;
	seqlen = 0;
	seqs = NULL;

	fstream fin;
	string tmpseq;
	
	// ambiguos characters
	meaning[0][0] = meaning[0][1] = 1; meaning[0][2] = meaning[0][3] =  meaning[0][4] =  0; // M
	meaning[1][0] = meaning[1][2] = 1; meaning[1][1] = meaning[1][3] =   meaning[1][4] = 0; // R
	meaning[2][0] = meaning[2][3] = 1; meaning[2][1] = meaning[2][2] =  meaning[2][4] = 0; // W
	meaning[3][1] = meaning[3][2] = 1; meaning[3][0] = meaning[3][3] =   meaning[3][4] = 0; // S
	meaning[4][1] = meaning[4][3] = 1; meaning[4][0] = meaning[4][2] =   meaning[4][4] = 0; // Y
	meaning[5][2] = meaning[5][3] = 1; meaning[5][0] = meaning[5][1] =   meaning[5][4] = 0; // K

	meaning[6][0] = meaning[6][1]  = meaning[6][2] = 1 ;  meaning[6][3] = meaning[6][4] =  0; // V
	meaning[7][0] = meaning[7][1]  = meaning[7][3] = 1 ;  meaning[7][2] = meaning[7][4] = 0; // H
	meaning[8][0] = meaning[8][2]  = meaning[8][3] = 1 ;  meaning[8][1] = meaning[8][4] = 0; // D
	meaning[9][1] = meaning[9][2]  = meaning[9][3] = 1 ;  meaning[9][0] = meaning[9][4] = 0; // B
	meaning[10][1] = meaning[10][2]  = meaning[10][3] =  meaning[10][0] =  0; meaning[10][4] = 1; // gap
	meaning[11][1] = meaning[11][2]  = meaning[11][3] =  meaning[11][0] = 1; meaning[11][4] = 0; // undefined
	// open the file for reading
	
	try{
		fin.open(fn, ios::in);
		if(!fin.is_open())
		{
			cout << "\n" << fn << endl;
			throw ExceptionManager(10);
		}
	
		// read numseq and seqlen	
		if(!(fin >> numseqs >> seqlen))
		{
			throw ExceptionManager(11);
		}
		seqs = new unsigned char[numseqs*seqlen];
		seqnames.resize(numseqs);
		
		for(int i=0; i < numseqs; i++)
		{
			fin >>  seqnames[i];
			fin >>  tmpseq;
			for(int j=0; j<seqlen; j++)
			{
				switch(toupper(tmpseq[j]))
				{
					case 'A': seqs[i*seqlen+j] = 0; break;
					case 'C': seqs[i*seqlen+j] = 1;  break;
					case 'G': seqs[i*seqlen+j] = 2; break;
					case 'T': seqs[i*seqlen+j] = 3; break;
					case 'M': seqs[i*seqlen+j] = 4; break;
					case 'R': seqs[i*seqlen+j] = 5; break;
					case 'W': seqs[i*seqlen+j] = 6; break;
					case 'S': seqs[i*seqlen+j] = 7; break;
					case 'Y': seqs[i*seqlen+j] = 8; break;
					case 'K': seqs[i*seqlen+j] = 9; break;
					case 'V': seqs[i*seqlen+j] = 10; break;
					case 'H': seqs[i*seqlen+j] = 11; break;
					case 'D': seqs[i*seqlen+j] = 12; break;
					case 'B': seqs[i*seqlen+j] = 13; break;
					case '-': seqs[i*seqlen+j] = 14; break; // gap character
					case '?':
					case 'X':
					case 'N' :
					case 'O' : seqs[i*seqlen+j] = 15; break; // undefined character
					default :
					{
						
						cout << "invalid character:" << tmpseq[j] << " seq:" << seqnames[i] << "  column " << j << endl;
						throw ExceptionManager(11);
						assert(1==0); break; // invalid character
					}
				}
			}
		}	
	}
	catch( ExceptionManager e)
	{
		e.Report();
	}
	fin.close();
}


int Sequences::search_taxon_name(std::string &s) const
{
	int i;
	for(i=numseqs-1; i>=0 && seqnames[i].substr(0,10)!=s; i--);
	return i;
}


// return true is the site is informative
bool Sequences::is_informative(unsigned int col) const
{
	unsigned char temp[16];
	int sum = 0;
	memset( temp, 0, 16*sizeof(unsigned char));
	for(int j=0; j< numseqs; j++)
	{
			// check if the site is repeated
		unsigned char l = (*this)[j][col]; //SeqData->pattern_pos(i,j);
		if( temp[l] == 0)
		{
			temp[l] = 1; sum++;
		}
	}
	// informative sites have at least two differents chars
	return(sum > 1);
}


void Sequences::calculate_patterns()
{
	// map pattern and index in the vector pattern
	num_patterns=num_inf_sites=0;
	
	map<string, int> pattern_map;
	struct PatternInfo p;
	string s;
	for(int i=0; i< seqlen; i++)
	{
		
		s = position(i);
		//if(s.find('?',0)== string::npos)cout << i << endl;
		// new pattern
		if( pattern_map.find(s) == pattern_map.end() )
		{
			pattern_map[s] = num_patterns;
			p.idx = i;
			p.count = 1;
			patterns.push_back(p);
			// calculate informative site
			if( is_informative(i) )
			{
				inf_sites.push_back(num_patterns);
				num_inf_sites++;
			}
			num_patterns++;
		}
		// old pattern
		else
			patterns[ pattern_map[s] ].count++;
	}		
	//cout << "Padrones reconhecidos:" << num_patterns << endl;
	//cout << "Sites informativos   :" << num_inf_sites << endl;
}

void Sequences::calculate_frequences()
{
	long num_undefined = 0; // number of undefined positions
	freqs[0] = freqs[1] = freqs[2] = freqs[3] = 0.25;
	double tmp[4];
	double divisor;
	// i don't know why most program do this
	for(int m=0; m<8; m++)
	{
		tmp[0] = tmp[1] = tmp[2] = tmp[3] = 0;
		for(int i=0; i< num_patterns; i++)
			for(int j=0; j< numseqs; j++)
			{
				
				unsigned char l = pattern_pos(i,j);
				
				if ( is_ambiguous(l) )
				{
					divisor = 0;
					for(int k=0; k<4; k++) 
						divisor += ambiguos_meaning(l)[k] * freqs[k];
					for(int k=0; k<4; k++)
						tmp[k] += 
						(ambiguos_meaning(l)[k]*freqs[k]/divisor)* 
						patterns[i].count; 
				}
				else if( is_undefined(l) || is_gap(l) )
					for(int k=0; k<4; k++) tmp[k] += freqs[k]*patterns[i].count;
				else 
					tmp[l] += patterns[i].count; 
			}
		for(int k=0; k<4; k++) freqs[k] = tmp[k] / (tmp[0]+tmp[1]+tmp[2]+tmp[3]);
	}
	// final frequences
	//for(int i=0; i<4; i++){
	//	cout << "frequencia de " << i << " -->:" << freqs[i] << endl;
	//}
	// cout << "suma de frequencias:" << freqs[0] + freqs[1] + freqs[2] + freqs[3] << endl;
}


// get the 
string Sequences::position(int pos)
{
	string column;
	unsigned char c;
	char nucleotide;
	for(int i=0; i < numseqs; i++)
	{
		c = seq_pos(i, pos);
		switch(c)
		{
			case 0: nucleotide = 'A'; break;
			case 1: nucleotide = 'C'; break;
			case 2: nucleotide = 'G'; break;
			case 3: nucleotide = 'T'; break;
			case 4: nucleotide = 'M'; break;
			case 5: nucleotide = 'R'; break;
			case 6: nucleotide = 'W'; break;
			case 7: nucleotide = 'S'; break;
			case 8: nucleotide = 'Y'; break;
			case 9: nucleotide = 'K'; break;
			case 10: nucleotide = 'V'; break;
			case 11: nucleotide = 'H'; break;
			case 12: nucleotide = 'D'; break;
			case 13: nucleotide = 'B'; break;
			case 14: nucleotide = '-'; break;
			case 15: nucleotide = '?'; break;
			default: assert(1==0); break;
		}
		column+=nucleotide;
	}
	return column;
}

void Sequences::save_seq_data(ostream &of=cout)
{
	of << "\nSequence Datafile Statistics\n";
	of << "----------------------------------------------\n";
	of << "Number of Sequences : " << this->numseqs << endl;
	of << "Sequence Length     : " << this->seqlen << endl;
    of << "Number of Patterns  : " << this->num_patterns << endl;
	of << "Frequency A         : " << this->freqs[0] << endl;
	of << "Frequency C         : " << this->freqs[1] << endl;
	of << "Frequency G         : " << this->freqs[2] << endl;
	of << "Frequency T         : " << this->freqs[3] << endl;
}
