// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "bitMove.h"

#ifndef __bitMove
#define __bitMove

#include <utility>
#include <move/moMove.h>

template <class EOT>
class BitMove : public moMove <EOT> {

public :

    typedef EOT EOType;

    unsigned bit;

    BitMove() {
        bit = 0;
    }

    BitMove(unsigned _bit) : bit(_bit) { }


    void operator () (EOT & chrom)
    {
        chrom[bit] = !chrom[bit];
    };

} ;

#endif
