#!/bin/sh

DIE=0

PROG=eo

(autoconf --version) < /dev/null > /dev/null 2>&1 ||
{
    echo
    echo "You must have autoconf installed to compile $PROG."
    DIE=1
}

#(libtool --version) < /dev/null > /dev/null 2>&1 ||
#{
#    echo
#    echo "You must have libtool installed to compile $PROG."
#    DIE=1
#}

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
automake
autoconf

# we want doc to be recompiled - and it keeps saying it's up to date!!!
touch doc/eo.cfg


# if test -z "$*"; then
#     echo "I am going to run ./configure with no arguments - if you wish"
#     echo "to pass any to it, please specify them on the $0 command line."
# fi
#
# ./configure "$@"

echo
echo "Now run 'configure' and 'make' to build $PROG."
echo
echo "If you have Doxygen installed, type 'make doc' to generate $PROG documentation."
echo

#echo "WARNING: Compiling all test programs can take some time."
#echo "But you don't have to: you can simply type"
#echo "                    'make lib'"
#echo "and then go in your application dir (or in the tutorial dir)"
#echo "and there type 'make'"
