# Microsoft Developer Studio Project File - Name="eo" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Static Library" 0x0104

CFG=eo - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "eo.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "eo.mak" CFG="eo - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "eo - Win32 Release" (based on "Win32 (x86) Static Library")
!MESSAGE "eo - Win32 Debug" (based on "Win32 (x86) Static Library")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "eo - Win32 Release"

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
# ADD CPP /nologo /W3 /GR /GX /O2 /I "../src" /I "eo/src" /D "WIN32" /D "NDEBUG" /D "_MBCS" /D "_LIB" /FD /c
# SUBTRACT CPP /YX
# ADD BASE RSC /l 0xc0a /d "NDEBUG"
# ADD RSC /l 0xc0a /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo
# ADD LIB32 /nologo

!ELSEIF  "$(CFG)" == "eo - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "eo___Win32_Debug"
# PROP BASE Intermediate_Dir "eo___Win32_Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Debug"
# PROP Intermediate_Dir "Debug"
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_MBCS" /D "_LIB" /YX /FD /GZ /c
# ADD CPP /nologo /W3 /Gm /GR /GX /ZI /Od /I "../src" /D "WIN32" /D "_DEBUG" /D "_MBCS" /D "_LIB" /FD /GZ /c
# SUBTRACT CPP /YX
# ADD BASE RSC /l 0xc0a /d "_DEBUG"
# ADD RSC /l 0xc0a /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo
# ADD LIB32 /nologo

!ENDIF 

# Begin Target

# Name "eo - Win32 Release"
# Name "eo - Win32 Debug"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=..\src\utils\eoFileMonitor.cpp
# End Source File
# Begin Source File

SOURCE=..\src\eoFunctorStore.cpp
# End Source File
# Begin Source File

SOURCE=..\src\utils\eoParser.cpp
# End Source File
# Begin Source File

SOURCE=..\src\eoPersistent.cpp
# End Source File
# Begin Source File

SOURCE=..\src\eoPrintable.cpp
# End Source File
# Begin Source File

SOURCE=..\src\utils\eoRNG.cpp
# End Source File
# Begin Source File

SOURCE=..\src\utils\eoState.cpp
# End Source File
# Begin Source File

SOURCE=..\src\utils\eoStdoutMonitor.cpp
# End Source File
# Begin Source File

SOURCE=..\src\utils\eoUpdater.cpp
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# Begin Source File

SOURCE=..\src\utils\compatibility.h
# End Source File
# Begin Source File

SOURCE=..\src\utils\eoData.h
# End Source File
# Begin Source File

SOURCE=..\src\utils\eoFileMonitor.h
# End Source File
# Begin Source File

SOURCE=..\src\utils\eoMonitor.h
# End Source File
# Begin Source File

SOURCE=..\src\eoObject.h
# End Source File
# Begin Source File

SOURCE=..\src\utils\eoParam.h
# End Source File
# Begin Source File

SOURCE=..\src\utils\eoParser.h
# End Source File
# Begin Source File

SOURCE=..\src\eoPersistent.h
# End Source File
# Begin Source File

SOURCE=..\src\eoPrintable.h
# End Source File
# Begin Source File

SOURCE=..\src\utils\eoRNG.h
# End Source File
# Begin Source File

SOURCE=..\src\utils\eoStat.h
# End Source File
# Begin Source File

SOURCE=..\src\utils\eoState.h
# End Source File
# Begin Source File

SOURCE=..\src\utils\eoUpdater.h
# End Source File
# End Group
# End Target
# End Project
