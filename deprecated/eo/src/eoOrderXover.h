// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-
//-----------------------------------------------------------------------------
// eoOrderXover.h
//-----------------------------------------------------------------------------

#ifndef eoOrderXover_h
#define eoOrderXover_h

//-----------------------------------------------------------------------------

#include <algorithm>
#include <utils/eoRNG.h>
#include <eoInit.h>

/**
 * apply orderXover on two chromosomes.
 * Example:
 * With 2 parents p1= [1,2,4,3,6,5], p2= [4,5,3,1,6,2]
 * It applies a random cut between ("|") two indices. Example: the second element and the fourth:
 * p1 : [1|2,4,3|6,5] and p2 : [4|5,3,1|6,2] generate two children:
 * c1 = [2,4,3,6,5,1] 2,4,3 taken from p1 and 6,5,1 from p2
 * c2 = [5,3,1,6,2,4] 5,3,1 taken from p2 and 6,2,4 from p1
 *
 * @ingroup Variators
 */
template<class Chrom> class eoOrderXover: public eoQuadOp<Chrom>
{
 public:
         /// The class name.
         virtual std::string className() const { return "eoOrderXover"; }

         /**
          * @return true if the chromosome has changed
          * @param _chrom1 The first chromosome which will be crossed with chrom2.
          * @param _chrom2 The second chromosome which will be crossed with chrom1.
          */
         bool operator()(Chrom& _chrom1, Chrom& _chrom2){

                 char direction=eo::rng.flip()? 1 : -1;
                 unsigned cut2= 1 + eo::rng.random(_chrom1.size());
                 unsigned cut1= eo::rng.random(cut2);
                 Chrom tmp1= _chrom1;
                 Chrom tmp2= _chrom2;

                 cross(tmp1, tmp2, _chrom1, direction, cut1, cut2);
                 cross(tmp2, tmp1, _chrom2, direction, cut1, cut2);

                 _chrom1.invalidate();
                 _chrom2.invalidate();

                 return true;
         }

 private:

         /**
          * @param _chrom1 The first parent chromosome.
          * @param _chrom2 The second parent chromosome.
          * @param _child The result chromosome.
          * @param _direction The direction of the OrderXover (left: -1 or right: 1)
      * @param _cut1 index of the first cut
      * @param _cut2 index of the second cut
          */
         void cross(Chrom& _chrom1, Chrom& _chrom2, Chrom& _child, char _direction, unsigned _cut1, unsigned _cut2){

                unsigned size, id=0, from=0;
                size= _chrom1.size();

                std::vector<bool> verif(size, false);

                for(unsigned i= _cut1; i<_cut2; i++){
                        _child[id++]= _chrom1[i];
                        verif[_chrom1[i] % size] = true;
                }

                while(_chrom2[from] != _child[_cut2 - 1])
                        from++;

                for(unsigned i=0; i<size; i++){
                        unsigned j= (_direction*i + from + size) % size;
                        if(!verif[_chrom2[j] % size]){
                                _child[id++]=_chrom2[j];
                                verif[_chrom2[j]%size]=true;
                        }
                }
         }

};
/** @example t-eoOrderXover.cpp
 */

#endif
