# AC_APPLICATIONS()
#
# Compile applications unless user requests not to do it.
AC_DEFUN([AC_APPLICATIONS],[dnl
  AC_ARG_ENABLE([applications],[  --enable-applications          build applications (default=yes)],
    [ case "${enableval}" in
      yes) applications=true ;;
      no) applications=false ;;
      *) AC_MSG_ERROR(bad value ${enableval} for applications option) ;;
    esac],
    [applications=true])
  if test "$applications" = "true"; then
    AM_CONDITIONAL([USE_APPLICATIONS], true)
  else
    AM_CONDITIONAL([USE_APPLICATIONS], false)
  fi
])



# AC_TUTORIAL()
#
# Compile tutorial unless user requests not to do it.
AC_DEFUN([AC_TUTORIAL],[dnl
  AC_ARG_ENABLE([tutorial],[  --enable-tutorial              build tutorial (default=yes)],
    [ case "${enableval}" in
      yes) tutorial=true ;;
      no) tutorial=false ;;
      *) AC_MSG_ERROR(bad value ${enableval} for tutorial option) ;;
    esac],
    [tutorial=true])
  if test "$tutorial" = "true"; then
    AM_CONDITIONAL([USE_TUTORIAL], true)
  else
    AM_CONDITIONAL([USE_TUTORIAL], false)
  fi
])



dnl Available from the GNU Autoconf Macro Archive at:
dnl http://www.gnu.org/software/ac-archive/htmldoc/ac_cxx_have_sstream.html
dnl
AC_DEFUN([AC_CXX_HAVE_SSTREAM],
[AC_CACHE_CHECK(whether the compiler has stringstream,
ac_cv_cxx_have_sstream,
[AC_REQUIRE([AC_CXX_NAMESPACES])
 AC_LANG_SAVE
 AC_LANG_CPLUSPLUS
 AC_TRY_COMPILE([#include <sstream>
#ifdef HAVE_NAMESPACES
using namespace std;
#endif
#if __GNUC__ == 2 
#if __GNUC_MINOR__ == 96
#error("force_error_for_2_96")
#endif
#endif],[stringstream message; message << "Hello"; return 0;
],
 ac_cv_cxx_have_sstream=yes, ac_cv_cxx_have_sstream=no)
 AC_LANG_RESTORE
])
if test "$ac_cv_cxx_have_sstream" = yes; then
  AC_DEFINE(HAVE_SSTREAM,,[define if the compiler has stringstream])
fi
])



dnl Available from the GNU Autoconf Macro Archive at:
dnl http://www.gnu.org/software/ac-archive/htmldoc/ac_cxx_namespaces.html
dnl
AC_DEFUN([AC_CXX_NAMESPACES],
[AC_CACHE_CHECK(whether the compiler implements namespaces,
ac_cv_cxx_namespaces,
[AC_LANG_SAVE
 AC_LANG_CPLUSPLUS
 AC_TRY_COMPILE([namespace Outer { namespace Inner { int i = 0; }}],
                [using namespace Outer::Inner; return i;],
 ac_cv_cxx_namespaces=yes, ac_cv_cxx_namespaces=no)
 AC_LANG_RESTORE
])
if test "$ac_cv_cxx_namespaces" = yes; then
  AC_DEFINE(HAVE_NAMESPACES,,[define if the compiler implements namespaces])
fi
])



dnl Available from the GNU Autoconf Macro Archive at:
dnl http://www.gnu.org/software/ac-archive/htmldoc/ac_cxx_have_numeric_limits.html
dnl
AC_DEFUN([AC_CXX_HAVE_NUMERIC_LIMITS],
[AC_CACHE_CHECK(whether the compiler has numeric_limits<T>,
ac_cv_cxx_have_numeric_limits,
[AC_REQUIRE([AC_CXX_NAMESPACES])
 AC_LANG_SAVE
 AC_LANG_CPLUSPLUS
 AC_TRY_COMPILE([#include <limits>
#ifdef HAVE_NAMESPACES
using namespace std;
#endif],[double e = numeric_limits<double>::epsilon(); return 0;],
 ac_cv_cxx_have_numeric_limits=yes, ac_cv_cxx_have_numeric_limits=no)
 AC_LANG_RESTORE
])
if test "$ac_cv_cxx_have_numeric_limits" = yes; then
  AC_DEFINE(HAVE_NUMERIC_LIMITS,,[define if the compiler has numeric_limits<T>])
fi
])
