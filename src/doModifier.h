#ifndef _doModifier_h
#define _doModifier_h

template < typename D >
class doModifier
{
public:
    virtual ~doModifier(){}

    typedef typename D::EOType EOType;
};

#endif // !_doModifier_h
