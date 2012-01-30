# NOTE: only add something here if it is really needed by EO

INCLUDE(CheckIncludeFile)
INCLUDE(CheckIncludeFiles)
INCLUDE(CheckSymbolExists)
INCLUDE(CheckFunctionExists)
INCLUDE(CheckLibraryExists)

CHECK_LIBRARY_EXISTS(m	cos	"/usr/lib"	HAVE_LIBM)

CHECK_INCLUDE_FILES(math.h	"math.h"	HAVE_MATH_H)
CHECK_INCLUDE_FILES(stdio.h	"stdio.h"	HAVE_STDIO_H)
CHECK_INCLUDE_FILES(stdlib.h	"stdlib.h"	HAVE_STDLIB_H)
CHECK_INCLUDE_FILES(string.h	"string.h"	HAVE_STRING_H)
CHECK_INCLUDE_FILES(strings.h	"strings.h"	HAVE_STRINGS_H)
CHECK_INCLUDE_FILES(malloc.h	"malloc.h"	HAVE_MALLOC_H)
CHECK_INCLUDE_FILES(limits.h	"limits.h"	HAVE_LIMITS_H)
CHECK_INCLUDE_FILES(unistd.h	"unistd.h"	HAVE_UNISTD_H)
CHECK_INCLUDE_FILES(stdint.h	"stdint.h"	HAVE_STDINT_H)


# Use check_symbol_exists to check for symbols in a reliable
# cross-platform manner.  It accounts for different calling
# conventions and the possibility that the symbol is defined as a
# macro.  Note that some symbols require multiple includes in a
# specific order.  Refer to the man page for each symbol for which a
# check is to be added to get the proper set of headers. Example :

#check_symbol_exists(asymbol	"symbole.h"	HAVE_SYMBOLE)
