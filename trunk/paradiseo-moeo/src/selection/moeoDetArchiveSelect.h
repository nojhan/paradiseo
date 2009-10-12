#ifndef MOEODETARCHIVESELECT_H
#define MOEODETARCHIVESELECT_H

#include <eoSelect.h>
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
		virtual void operator()(const eoPop < MOEOT > & _source, eoPop < MOEOT > & _dest)
		{
			if(max < min){
				std::cout << "Warning! moeoDetArchiveSelect: min value > max value!!! Nothing is done." << std::endl;
			}
			else{
				unsigned int archive_size = archive.size();
				_dest.resize(0);
				if (archive_size >= min && archive_size<= max){
					for (int i=0; i<archive_size; i++)
						_dest.push_back(archive[i]);
				}
				else if (archive_size > max){
					UF_random_generator<unsigned int> rndGen;
					std::vector <int> permutation;
					for(int i=0; i < archive_size; i++)
						permutation.push_back(i);
					random_shuffle(permutation.begin(), permutation.end(), rndGen);
					for (int i=0; i<max; i++)
						_dest.push_back(archive[permutation[i]]);
				}
				else {
					for (int i=0; i<min; i++)
						_dest.push_back(archive[i%archive_size]);
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
