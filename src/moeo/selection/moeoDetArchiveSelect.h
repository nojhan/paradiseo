/*
* <moeoDetArchiveSelect.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Arnaud Liefooghe
* Jérémie Humeau
* François Legillon
* Thibaut demaret
*
* This software is governed by the CeCILL license under French law and
* abiding by the rules of distribution of free software.  You can  use,
* modify and/ or redistribute the software under the terms of the CeCILL
* license as circulated by CEA, CNRS and INRIA at the following URL
* "http://www.cecill.info".
*
* As a counterpart to the access to the source code and  rights to copy,
* modify and redistribute granted by the license, users are provided only
* with a limited warranty  and the software's author,  the holder of the
* economic rights,  and the successive licensors  have only  limited liability.
*
* In this respect, the user's attention is drawn to the risks associated
* with loading,  using,  modifying and/or developing or reproducing the
* software by the user in light of its specific status of free software,
* that may mean  that it is complicated to manipulate,  and  that  also
* therefore means  that it is reserved for developers  and  experienced
* professionals having in-depth computer knowledge. Users are therefore
* encouraged to load and test the software's suitability as regards their
* requirements in conditions enabling the security of their systems and/or
* data to be ensured and,  more generally, to use and operate it in the
* same conditions as regards security.
* The fact that you are presently reading this means that you have had
* knowledge of the CeCILL license and that you accept its terms.
*
* ParadisEO WebSite : http://paradiseo.gforge.inria.fr
* Contact: paradiseo-help@lists.gforge.inria.fr
*
*/
//-----------------------------------------------------------------------------

#ifndef MOEODETARCHIVESELECT_H
#define MOEODETARCHIVESELECT_H

#include "../../eo/eoSelect.h"
#include <cassert>

template<class MOEOT>
class moeoDetArchiveSelect : public eoSelect<MOEOT>
{
	public:

		moeoDetArchiveSelect(moeoArchive < MOEOT > & _archive, unsigned int _max, unsigned int _min=0) :
			archive(_archive), max(_max), min(_min) {}

		/**
		 * Repopulate copying the archive, selecting randomly if the archive size is over bounds (min and max)
		 * @param _source compatibility parameter, not used
		 * @param _dest destination population, selected from archive
		 */
		void operator()(const eoPop < MOEOT > & _source, eoPop < MOEOT > & _dest)
		{
			if(max < min){
				std::cout << "Warning! moeoDetArchiveSelect: min value > max value!!! Nothing is done." << std::endl;
			}
			else{

				unsigned int archive_size = archive.size();
				_dest.resize(0);
				if ((archive_size >= min) && (archive_size <= max)){
					for (unsigned int i=0; i<archive_size; i++)
						_dest.push_back(archive[i]);
				}
				else if (archive_size > max){
					UF_random_generator<unsigned int> rndGen;
					std::vector <int> permutation;
					for(unsigned int i=0; i < archive_size; i++)
						permutation.push_back(i);
					random_shuffle(permutation.begin(), permutation.end(), rndGen);
					for (unsigned int i=0; i<max; i++)
						_dest.push_back(archive[permutation[i]]);
				}
				else {
					for (unsigned int i=0; i<min; i++){
						_dest.push_back(archive[i%archive_size]);
					}
				}
			}
		}

	private :
		/** archive used to select MOEOT*/
		moeoArchive < MOEOT > & archive;
		/** max is the maximum size of the new population*/
		unsigned int max;
		/** min is the minimum size of the new population (default 0) */
		unsigned int min;
};

#endif /*MOEODETARCHIVESELECT_H*/
