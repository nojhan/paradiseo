# Microsoft Developer Studio Project File - Name="es" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Static Library" 0x0104

CFG=es - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "es.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "es.mak" CFG="es - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "es - Win32 Release" (based on "Win32 (x86) Static Library")
!MESSAGE "es - Win32 Debug" (based on "Win32 (x86) Static Library")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "es - Win32 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "Release"
# PROP Intermediate_Dir "Release"
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_MBCS" /D "_LIB" /YX /FD /c
# ADD CPP /nologo /w /W0 /GR /GX /O2 /I "..\src" /D "WIN32" /D "NDEBUG" /D "_MBCS" /D "_LIB" /D "NO_GNUPLOT" /YX /FD /Zm200 /c
# ADD BASE RSC /l 0x40c /d "NDEBUG"
# ADD RSC /l 0x40c /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo
# ADD LIB32 /nologo
# Begin Special Build Tool
SOURCE="$(InputPath)"
PostBuild_Desc=Install
PostBuild_Cmds=copy  Release\es.lib  lib\ 
# End Special Build Tool

!ELSEIF  "$(CFG)" == "es - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Debug"
# PROP Intermediate_Dir "Debug"
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_MBCS" /D "_LIB" /YX /FD /GZ /c
# ADD CPP /nologo /w /W0 /Gm /GR /GX /ZI /Od /I "..\src" /D "WIN32" /D "_DEBUG" /D "_MBCS" /D "_LIB" /D "NO_GNUPLOT" /YX /FD /GZ /Zm200 /c
# ADD BASE RSC /l 0x40c /d "_DEBUG"
# ADD RSC /l 0x40c /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo
# ADD LIB32 /nologo /out:"Debug\esd.lib"
# Begin Special Build Tool
SOURCE="$(InputPath)"
PostBuild_Desc=Install
PostBuild_Cmds=copy  Debug\esd.lib  lib\ 
# End Special Build Tool

!ENDIF 

# Begin Target

# Name "es - Win32 Release"
# Name "es - Win32 Debug"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=..\src\es\make_algo_scalar_es.cpp
# End Source File
# Begin Source File

SOURCE=..\src\es\make_algo_scalar_real.cpp
# End Source File
# Begin Source File

SOURCE=..\src\es\make_checkpoint_es.cpp
# End Source File
# Begin Source File

SOURCE=..\src\es\make_checkpoint_real.cpp
# End Source File
# Begin Source File

SOURCE=..\src\es\make_continue_es.cpp
# End Source File
# Begin Source File

SOURCE=..\src\es\make_continue_real.cpp
# End Source File
# Begin Source File

SOURCE=..\src\es\make_genotype_es.cpp
# End Source File
# Begin Source File

SOURCE=..\src\es\make_genotype_real.cpp
# End Source File
# Begin Source File

SOURCE=..\src\es\make_op_es.cpp
# End Source File
# Begin Source File

SOURCE=..\src\es\make_op_real.cpp
# End Source File
# Begin Source File

SOURCE=..\src\es\make_pop_es.cpp
# End Source File
# Begin Source File

SOURCE=..\src\es\make_pop_real.cpp
# End Source File
# Begin Source File

SOURCE=..\src\es\make_run_es.cpp
# End Source File
# Begin Source File

SOURCE=..\src\es\make_run_real.cpp
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# Begin Source File

SOURCE=..\src\es\eoEsChromInit.h
# End Source File
# Begin Source File

SOURCE=..\src\es\eoEsFull.h
# End Source File
# Begin Source File

SOURCE=..\src\es\eoEsGlobalXover.h
# End Source File
# Begin Source File

SOURCE=..\src\es\eoEsMutate.h
# End Source File
# Begin Source File

SOURCE=..\src\es\eoEsMutationInit.h
# End Source File
# Begin Source File

SOURCE=..\src\es\eoEsSimple.h
# End Source File
# Begin Source File

SOURCE=..\src\es\eoEsStandardXover.h
# End Source File
# Begin Source File

SOURCE=..\src\es\eoEsStdev.h
# End Source File
# Begin Source File

SOURCE=..\src\es\eoNormalMutation.h
# End Source File
# Begin Source File

SOURCE=..\src\es\eoReal.h
# End Source File
# Begin Source File

SOURCE=..\src\es\eoRealAtomXover.h
# End Source File
# Begin Source File

SOURCE=..\src\es\eoRealInitBounded.h
# End Source File
# Begin Source File

SOURCE=..\src\es\eoRealOp.h
# End Source File
# Begin Source File

SOURCE=..\src\es\make_es.h
# End Source File
# Begin Source File

SOURCE=..\src\es\make_genotype_real.h
# End Source File
# Begin Source File

SOURCE=..\src\es\make_op.h
# End Source File
# Begin Source File

SOURCE=..\src\es\make_op_es.h
# End Source File
# Begin Source File

SOURCE=..\src\es\make_op_real.h
# End Source File
# Begin Source File

SOURCE=..\src\es\make_real.h
# End Source File
# End Group
# End Target
# End Project
ÿ