#ifndef _moContinuator_h
#define _moContinuator_h

/*
  to make specific continuator from a solution
*/
template< class NH >
class moContinuator : public eoUF<typename NH::EOT &, bool>
{
public:
    typedef NH Neighborhood ;
    typedef typename Neighborhood::EOT EOT ;

    // empty constructor
    moContinuator() { } ;

    virtual void init(EOT & solution) = 0 ;
};

#endif


// Local Variables:
// coding: iso-8859-1
// mode: C++
// c-file-offsets: ((c . 0))
// c-file-style: "Stroustrup"
// fill-column: 80
// End:
