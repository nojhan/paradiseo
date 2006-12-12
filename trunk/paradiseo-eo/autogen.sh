#!/bin/sh

DIE=0
PROG=eo

(autoconf --version) < /dev/null > /dev/null 2>&1 ||
{
    echo
    echo "You must have autoconf installed to compile $PROG."
    DIE=1
}

(automake --version) < /dev/null > /dev/null 2>&1 ||
{
    echo
    echo "You must have automake installed to compile $PROG."
    DIE=1
}

if test "$DIE" -eq 1; then
    exit 1
fi

set aclocalinclude="$ACLOCAL_FLAGS"
aclocal $aclocalinclude
unset $aclocalinclude
autoheader
automake -a -c
autoconf

# we want doc to be recompiled - and it keeps saying it's up to date!!!
# touch doc/eo.cfg

echo
echo "Now run 'configure' and 'make' to build $PROG."
echo "You can check the libraries by running 'make check'"
echo
echo "If you have Doxygen installed, type 'make doc' to generate $PROG documentation."
echo
