# NOTE: only add something here if it is really needed by EO

include(CheckIncludeFile)
include(CheckIncludeFiles)
include(CheckSymbolExists)
include(CheckFunctionExists)
include(CheckLibraryExists)

check_library_exists(m   cos "cos in libm" HAVE_LIBM)

check_include_files(cmath.h  "cmath.h"      HAVE_CMATH_H)
check_include_files(stdio.h  "stdio.h"      HAVE_STDIO_H)
check_include_files(stdlib.h "stdlib.h"     HAVE_STDLIB_H)
check_include_files(string.h  "string.h"    HAVE_STRING_H)
check_include_files(strings.h "strings.h"   HAVE_STRINGS_H)
check_include_files(malloc.h  "malloc.h"    HAVE_MALLOC_H)
check_include_files(limits.h  "limits.h"    HAVE_LIMITS_H)
check_include_files(unistd.h  "unistd.h"    HAVE_UNISTD_H)
check_include_files(stdint.h  "stdint.h"    HAVE_STDINT_H)


# Use check_symbol_exists to check for symbols in a reliable
# cross-platform manner.  It accounts for different calling
# conventions and the possibility that the symbol is defined as a
# macro.  Note that some symbols require multiple includes in a
# specific order.  Refer to the man page for each symbol for which a
# check is to be added to get the proper set of headers. Example :

#check_symbol_exists(asymbol          "symbole.h"                 HAVE_SYMBOLE)
