# Microsoft Developer Studio Project File - Name="ga" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Static Library" 0x0104

CFG=ga - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "ga.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "ga.mak" CFG="ga - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "ga - Win32 Release" (based on "Win32 (x86) Static Library")
!MESSAGE "ga - Win32 Debug" (based on "Win32 (x86) Static Library")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "ga - Win32 Release"

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
# ADD CPP /nologo /MT /w /W0 /GR /GX /O2 /I "..\src" /D "WIN32" /D "NDEBUG" /D "_MBCS" /D "_LIB" /D "NO_GNUPLOT" /YX /FD /c
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
PostBuild_Cmds=copy   Release\ga.lib   lib\ 
# End Special Build Tool

!ELSEIF  "$(CFG)" == "ga - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "ga___Win32_Debug"
# PROP BASE Intermediate_Dir "ga___Win32_Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Debug"
# PROP Intermediate_Dir "Debug"
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_MBCS" /D "_LIB" /YX /FD /GZ /c
# ADD CPP /nologo /MTd /w /W0 /Gm /GR /GX /ZI /Od /I "..\src" /D "WIN32" /D "_DEBUG" /D "_MBCS" /D "_LIB" /D "NO_GNUPLOT" /YX /FD /GZ /c
# ADD BASE RSC /l 0x40c /d "_DEBUG"
# ADD RSC /l 0x40c /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo
# ADD LIB32 /nologo /out:"Debug\gad.lib"
# Begin Special Build Tool
SOURCE="$(InputPath)"
PostBuild_Desc=Install
PostBuild_Cmds=copy   Debug\gad.lib   lib\ 
# End Special Build Tool

!ENDIF 

# Begin Target

# Name "ga - Win32 Release"
# Name "ga - Win32 Debug"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=..\src\ga\make_algo_scalar_ga.cpp
# End Source File
# Begin Source File

SOURCE=..\src\ga\make_checkpoint_ga.cpp
# End Source File
# Begin Source File

SOURCE=..\src\ga\make_continue_ga.cpp
# End Source File
# Begin Source File

SOURCE=..\src\ga\make_genotype_ga.cpp
# End Source File
# Begin Source File

SOURCE=..\src\ga\make_op_ga.cpp
# End Source File
# Begin Source File

SOURCE=..\src\ga\make_pop_ga.cpp
# End Source File
# Begin Source File

SOURCE=..\src\ga\make_run_ga.cpp
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# Begin Source File

SOURCE=..\src\ga\eoBit.h
# End Source File
# Begin Source File

SOURCE=..\src\ga\eoBitOp.h
# End Source File
# Begin Source File

SOURCE=..\src\ga\eoBitOpFactory.h
# End Source File
# Begin Source File

SOURCE=..\src\ga\make_ga.h
# End Source File
# Begin Source File

SOURCE=..\src\ga\make_genotype_ga.h
# End Source File
# Begin Source File

SOURCE=..\src\ga\make_op.h
# End Source File
# End Group
# End Target
# End Project
