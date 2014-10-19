/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

The above line is usefulin Emacs-like editors
 */

/*
Template for creating a new representation in EO
================================================
*/

#ifndef _eoOneMax_h
#define _eoOneMax_h

/**
 *  Always write a comment in this format before class definition
 *  if you want the class to be documented by Doxygen

 * Note that you MUST derive your structure from EO<fitT>
 * but you MAY use some other already prepared class in the hierarchy
 * like eoVector for instance, if you handle a vector of something....

 * If you create a structure from scratch,
 * the only thing you need to provide are
 *        a default constructor
 *        IO routines printOn and readFrom
 *
 * Note that operator<< and operator>> are defined at EO level
 * using these routines
 */
template< class FitT>
class eoOneMax: public EO<FitT> {
public:
  /** Ctor: you MUST provide a default ctor.
   * though such individuals will generally be processed
   * by some eoInit object
   */
  eoOneMax()
  {
    // START Code of default Ctor of an eoOneMax object
    // END   Code of default Ctor of an eoOneMax object
  }

  virtual ~eoOneMax()
  {
    // START Code of Destructor of an eoEASEAGenome object
    // END   Code of Destructor of an eoEASEAGenome object
  }

  virtual string className() const { return "eoOneMax"; }

  /** printing... */
    void printOn(ostream& _os) const
    {
      // First write the fitness
      EO<FitT>::printOn(_os);
      _os << ' ';
    // START Code of default output

	/** HINTS
	 * in EO we systematically write the sizes of things before the things
	 * so readFrom is easier to code (see below)
	 */
      _os << b.size() << ' ' ;
      for (unsigned i=0; i<b.size(); i++)
	_os << b[i] << ' ' ;
    // END   Code of default output
    }

  /** reading...
   * of course, your readFrom must be able to read what printOn writes!!!
   */
  void readFrom(istream& _is)
  {
    // of course you should read the fitness first!
    EO<FitT>::readFrom(_is);
    // START Code of input

    /** HINTS
     * remember the eoOneMax object will come from the default ctor
     * this is why having the sizes written out is useful
     */
    unsigned s;
    _is >> s;
    b.resize(s);
    for (unsigned i=0; i<s; i++)
      {
	bool bTmp;
	_is >> bTmp;
	b[i] = bTmp;
      }
    // END   Code of input
  }

  // accessing and setting values
  void setB(vector<bool> & _b)
  {
    b=_b;
  }
  const vector<bool> & B()
  {
    return b;
  }

private:			   // put all data here
    // START Private data of an eoOneMax object
  std::vector<bool> b;
    // END   Private data of an eoOneMax object
};

#endif
