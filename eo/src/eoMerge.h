//-----------------------------------------------------------------------------
// eoMerge.h
//-----------------------------------------------------------------------------

#ifndef eoMerge_h
#define eoMerge_h

//-----------------------------------------------------------------------------

#include <eoPopOps.h>  // eoMerge

//-----------------------------------------------------------------------------
// eoInsertion
//-----------------------------------------------------------------------------

template<class Chrom> class Insertion: public eoMerge<Chrom>
{
 public:
  eoInsertion(const float& _rate = 1): eoMerge<Chrom>(rate) {}
  
  bool compare(const Chrom& chrom1, const Chrom& chrom2)
    {
      return chrom1.fitness() < chrom2.fitness();
    }
  
  void operator()(const Pop& breeders, Pop& pop)
    {    
      sort(pop.begin(), pop.end() compare);
      
      pop.erase(pop.end() + (int)(pop.size() * (rate - 1) - breeders.size()), 
		pop.end());
      
      copy(breeders.begin(), breeders.end(), 
	   back_insert_iterator<Pop>(pop));
    }
};

//-----------------------------------------------------------------------------

template<class Fitness> class Inclusion: public Replace<Fitness>
{
 public:
  Inclusion(const float& rate = 1): Replace<Fitness>(rate) {}
  
  void operator()(Pop& breeders, Pop& pop)
    {
      Pop temp;
      
      sort(breeders.begin(), breeders.end(), compare);
      sort(pop.begin(), pop.end(), compare);
      
      merge(breeders.begin(), breeders.end(), 
	    pop.begin(), pop.end(), 
	    back_insert_iterator<Pop>(temp), compare);
      
      temp.erase(temp.begin() + (unsigned)(rate * pop.size()), temp.end());
      pop.swap(temp);
    }
};

//-----------------------------------------------------------------------------

#endif eoMerge_h
