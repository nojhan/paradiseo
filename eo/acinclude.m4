# AC_APPLICATIONS()
#
# Compile applications unless user requests not to do it.
AC_DEFUN([AC_APPLICATIONS],[dnl
  AC_ARG_ENABLE([applications],
    AC_HELP_STRING([--enable-applications], [build applications (default=yes)]),
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



# AC_DEBUG()
#
# Check for commandline options requesting DEBUG feature.
# Might define DEBUG, NODEBUG, or NDEBUG, depending on flags given.
AC_DEFUN([AC_DEBUG],[dnl
  AC_ARG_ENABLE([debug], 
    AC_HELP_STRING([--enable-debug], [include debugging code (default=no)]),
    [ case "${enableval}" in
      yes) debug=true ;;
      no) debug=false ;;
      *) AC_MSG_ERROR(bad value ${enableval} for debug option) ;;
    esac],
    [debug=false])
  if test "$debug" = true; then
    AC_DEFINE([DEBUG], true, [Debugging flag])
  else
    AC_DEFINE([NDEBUG], true, [no debugging])
    AC_DEFINE([NODEBUG], true, [no debugging])
  fi
])



# AC_GNUPLOT()
#
# Compile applications unless user requests not to do it.
AC_DEFUN([AC_GNUPLOT], [dnl
    AC_ARG_ENABLE([gnuplot],
                  AC_HELP_STRING([--enable-gnuplot], [use gnuplot for graphical display (default=yes)]),
                  [ac_cv_use_gnuplot=$enableval],
                  [ac_cv_use_gnuplot=yes])
    AC_CACHE_CHECK([use gnuplot for graphical display],
                   [ac_cv_use_gnuplot],
                   [ac_cv_use_gnuplot=no])
    if test "$ac_cv_use_gnuplot" = "yes"; then
        AC_ARG_VAR([GNUPLOT], [gnuplot executable used for graphical display])
        AC_CHECK_PROG([GNUPLOT], [gnuplot], [gnuplot], [true])
        AC_DEFINE([HAVE_GNUPLOT], [1], [gnuplot graphical display])
    else
        AC_DEFINE([NO_GNUPLOT], [1], [no gnuplot graphical display -- deprecated, will be reomoved!])
    fi
])



# AC_TUTORIAL()
#
# Compile tutorial unless user requests not to do it.
AC_DEFUN([AC_TUTORIAL],[dnl
  AC_ARG_ENABLE([tutorial],
    AC_HELP_STRING([--enable-tutorial], [build tutorial (default=yes)]),
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
