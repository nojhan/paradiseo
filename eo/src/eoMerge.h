//-----------------------------------------------------------------------------
// eoMerge.h
//-----------------------------------------------------------------------------

#ifndef eoMerge_h
#define eoMerge_h

//-----------------------------------------------------------------------------

#include <eoPop.h>  // eoPop

//-----------------------------------------------------------------------------

/** eoMerge involves three populations, that can be merged and transformed to
give a third
*/
template<class EOT>
class eoMerge: public eoObject{

 public:
  /// (Default) Constructor.
  eoMerge(const float& _rate = 1.0): rep_rate(_rate) {}
  
  /// Dtor
  virtual ~eoMerge() {}
  
  /** Pure virtual transformation function. Extracts something from breeders
   *  and transfers it to the pop
   *  @param breeders Tranformed individuals.
   *  @param pop The original population at the begining, the result at the end
   */
  virtual void operator () ( eoPop<EOT>& breeders, eoPop<EOT>& pop ) = 0;
  
  /** @name Methods from eoObject	*/
  //@{
  /** readFrom and printOn are not overriden
   */
  /** Inherited from eoObject. Returns the class name.
      @see eoObject
  */
  string className() const {return "eoMerge";};
  //@}
  
  /// Return the rate to be selected from the original population
  float rate() const { return rep_rate; }

  /// Set the rate to be obtained after replacement.
  /// @param _rate The rate.
  void rate(const float& _rate) { rep_rate = _rate; }
  
private:
  float rep_rate;
};

//-----------------------------------------------------------------------------

#endif eoMerge_h
