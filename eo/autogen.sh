#!/bin/sh

DIE=0

PROG=eo

(autoconf --version) < /dev/null > /dev/null 2>&1 ||
{
    echo
    echo "You must have autoconf installed to compile $PROG."
    DIE=1
}

(libtool --version) < /dev/null > /dev/null 2>&1 ||
{
    echo
    echo "You must have libtool installed to compile $PROG."
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

if test -z "$*"; then
    echo "I am going to run ./configure with no arguments - if you wish"
    echo "to pass any to it, please specify them on the $0 command line."
fi

for dir in .
do
    echo processing $dir
    (
	cd $dir; \
	aclocalinclude="$ACLOCAL_FLAGS"; \
	aclocal $aclocalinclude; \
	autoheader; \
	automake; \
	autoconf
    )
done

./configure "$@"

echo
echo "Now type 'make' to compile $PROG."
