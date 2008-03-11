; ParadisEO install script
; Author: Thomas Legrand

#define cmakeLookupWizardPageIndex= 8
#define generatorWizardPageIndex= 8
#define buildModeWizardPageIndex= 9
#define launchBuildWizardPageIndex= 12

#define ctestMemCheckDummyError= 16
#define dartSubmissionError= 64

//***************************************************************************************//
//********** ParadisEO Specific strategy - The rest of the code is generic ************* //
//***************************************************************************************//

// where will the installer be created ?
#define OutputPath="E:\software\paradisEO\windows installer\compiler output"

// installer source info
//#define ParadiseoSourceTag="E:\software\paradisEO\repository\tags\paradiseo-ix86-1.0\*"
//#define EoTag="E:\software\eo\repository\tag_v_peo_1_0\*"
#define ParadiseoSourceTag="E:\software\paradisEO\repository\trunk\*"
#define EoTag="E:\software\eo\repository\eo-ROOT\*"

// installer description info
#define Version="1.1"
#define VersionMain="ParadisEO-ix86-1.1"
#define InstallerName="paradiseo-1.1-win32-installer"
#define ApplicationName="ParadisEO"
#define SetupIconPath="E:\software\paradisEO\repository\utilities\trunk\windows\img\paradiseo.ico"
#define WizardMainImage="E:\software\paradisEO\repository\utilities\trunk\windows\img\paradiseo.bmp"
#define SkipParadiseoFiles="lib,installParadisEO.sh,paradiseo-peo,.mpd.conf"
#define LicenceFile="E:\software\paradisEO\repository\trunk\LICENSE"
#define FinalInstructions="E:\software\paradisEO\repository\utilities\trunk\windows\final_instructions.txt"

// additionnal info
#define Company="INRIA"
#define AboutUS="INRIA Futurs Dolphin Project-team"
#define PublisherURL="http://paradiseo.gforge.inria.fr"
#define SupportURL="http://paradiseo.gforge.inria.fr"
#define UpdatesURL="http://paradiseo.gforge.inria.fr"

// logger
#define InstallLogger="logger "

//***************************************************************************************//

[Setup]
AppName={#ApplicationName}
AppVerName={#VersionMain}
AppPublisher={#AboutUS}
AppPublisherURL={#PublisherURL}
AppSupportURL={#SupportURL}
AppUpdatesURL={#UpdatesURL}
DefaultDirName={pf}\{#ApplicationName}
DefaultGroupName={#ApplicationName}
LicenseFile={#LicenceFile}
OutputDir={#OutputPath}
OutputBaseFilename={#InstallerName}
Compression=lzma/max
SolidCompression=yes
WizardImageFile={#WizardMainImage}
SetupIconFile={#SetupIconPath}
UninstallDisplayName={#ApplicationName}
WindowVisible=False
RestartIfNeededByRun=False
ShowTasksTreeLines=True
VersionInfoVersion={#Version}
VersionInfoCompany={#Company}
VersionInfoDescription={#ApplicationName}
VersionInfoTextVersion={#ApplicationName}
SetupLogging=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "french"; MessagesFile: "compiler:Languages\French.isl"

[CustomMessages]
english.CMakeMissing=CMake cannot be found on your computer. You must have CMake installed to use ParadisEO.
english.FullInstall=Full installation
english.CustomInstall=Custom installation
english.EoDescription=EO: Evolving Objects: Library for evolutionnary computation
english.MoDescription=MO: Moving Objects: Single based metaheuristics computation
english.MoeoDescription=MOEO: Multi Objective Evolving Objects
english.ErrorOccured=An error has occured
english.LaunchingBuildProcess=Launching CMake build process and the compilation...
english.LaunchingEOBuildProcess=Configuring ParadisEO-EO...
english.LaunchingMOBuildProcess=Configuring ParadisEO-MO...
english.LaunchingMOEOBuildProcess=Configuring ParadisEO-MOEO...
english.LaunchingEOCompilation=Compiling ParadisEO-EO...
english.LaunchingMOCompilation=Compiling ParadisEO-MO...
english.LaunchingMOEOCompilation=Compiling ParadisEO-MOEO...
english.error=Error
english.ErrorAbort=Error,abort.
english.CannotCompleteInstall=Impossible to complete the install of
english.BPFinished=Finished
english.BPSuccessfull=The installation has been successfully performed.
english.SelectCompiler=Select the program you want to use to compile:
english.ChooseGeneratorTitle=ParadisEO compilation
english.ChooseGeneratorSubtitle=Compiler selection
english.GenCMakeFiles=CMake configuration
english.BuildProcess=CMake configuration files generation and compilation
english.NextGenCaption=Click on the 'Next' button to launch CMake and compile ...
english.ProcessingCMake=Configuration and compilation
english.DolphinMsg=ParadisEO: An INRIA Dolphin Team project - Program developped by Thomas Legrand
english.BuildMode=Build and compilation mode.
english.SelectBuildMode=Please select the build and compilation mode:
english.Recommended=(recommended)
english.AcceptSendReport=I agree that the installation report will be sent to the DOLPHIN Team.
english.NoInfoSend1=Neither personal information nor data refering your computer will be sent.
english.NoInfoSend2=I could get a personalized and adapted support.
english.PathToCMakeTitle=Path to CMake
english.PathToCMakeSubTitle=CMake has not been found by the assistant. Please select the directory where CMake is installed on your computer
english.CMakeNotFound=The CMake binaries cannot be found in this directory
english.CMakeDownloadMsg=CMake available for download at:
english.NextGenCaptionPgmBegin= Notice that the generator you chose must be installed on your computer.

french.CMakeMissing=CMake n'a pas �t� d�tect� sur votre ordinateur. CMake doit �tre install� pour utiliser ParadisEO.
french.FullInstall=Installation compl�te
french.CustomInstall=Installation personnalis�e
french.EoDescription= EO: Evolving Objects: Librairie d�di�e aux m�thodes �volutionnaires
french.MoDescription= MO: Moving Objects: M�taheuristiques � base de solutions uniques
french.MoeoDescription= MOEO: Multi Objective Evolving Objects: Module multi-objectif
french.ErrorOccured=Une erreur est survenue
french.LaunchingBuildProcess=Construction des fichiers de configuration...
french.LaunchingEOBuildProcess=Configuration de ParadisEO-EO...
french.LaunchingMOBuildProcess=Configuration de ParadisEO-MO...
french.LaunchingMOEOBuildProcess=Configuration de ParadisEO-MOEO...
french.LaunchingEOCompilation=Compilation de ParadisEO-EO...
french.LaunchingMOCompilation=Compilation de ParadisEO-MO...
french.LaunchingMOEOCompilation=Compilation de ParadisEO-MOEO..
french.error=Erreur
french.ErrorAbort=Une erreur est survenue, installation annul�e.
french.CannotCompleteInstall=Impossible de terminer l'installation de
french.BPFinished=Fin de l'installation
french.BPSuccessfull=Succ�s.
french.SelectCompiler=S�lectionnez le programme que vous souhaitez utiliser pour compiler:
french.ChooseGeneratorTitle=Compilation de ParadisEO
french.ChooseGeneratorSubtitle=Selection du compilateur � utiliser
french.GenCMakeFiles=Configuration CMake
french.BuildProcess=G�n�ration des fichiers de configuration CMake et compilation
french.NextGenCaption=Cliquez sur le bouton 'Suivant' pour lancer CMake et compiler.
french.ProcessingCMake=Configuration et compilation
french.DolphinMsg=ParadisEO: Un projet de l'�quipe INRIA Dolphin - Programme r�alis� par Thomas Legrand
french.BuildMode=Choix du mode de compilation.
french.SelectBuildMode=Veuillez s�lectionner le type de compilation :
french.Recommended=(recommand�)
french.AcceptSendReport=Je souhaite que le rapport d'installation soit envoy� � l'�quipe DOLPHIN.
french.NoInfoSend1=Aucune information personnelle ou li�e � mon poste de travail ne sera transmise.
french.NoInfoSend2=Je pourrais b�n�ficier d'un support personnalis� et adapt�.
french.PathToCMakeTitle=Chemin vers CMake
french.PathToCMakeSubTitle=CMake n'a pas �t� trouv� par l'assistant. Veuillez s�lectionner le r�pertoire d'installation de CMake sur votre ordinateur
french.CMakeNotFound=Les ex�cutables CMake sont introuvables dans ce r�pertoire
french.CMakeDownloadMsg=CMake t�l�chargeable � l'adresse:
french.NextGenCaptionPgmBegin= Notez que le g�n�rateur que vous avez s�lectionn� doit �tre install� sur votre ordinateur.


[Types]
Name: "custom"; Description: {cm:CustomInstall}; Flags: iscustom
Name: "full"; Description: {cm:FullInstall}

[Components]
Name: eo; Description: {cm:EoDescription}; Types: full custom; Flags: fixed
Name: mo; Description:{cm:MoDescription}; Types: full custom;
Name: moeo; Description: {cm:MoeoDescription}; Types: full custom;


[Files]
Source: {#ParadiseoSourceTag}; DestDir: "{app}"; Excludes: {#SkipParadiseoFiles} ; Flags: ignoreversion recursesubdirs createallsubdirs
Source: {#EoTag}; DestDir: "{app}";  Excludes: "*.~*" ; Flags: ignoreversion recursesubdirs createallsubdirs

; SPECIFIC CORRECTION - EO TAG NOT MODIFIED, USED TRUNK SOURCES
Source: E:\software\eo\repository\eo-ROOT\paradiseo-eo\CMakeLists.txt ; DestDir: "{app}\paradiseo-eo";  Excludes: "*.~*" ; Flags: ignoreversion recursesubdirs createallsubdirs
Source: E:\software\eo\repository\eo-ROOT\paradiseo-eo\src\utils\pipecom.h; DestDir: "{app}\paradiseo-eo\src\utils";  Excludes: "*.~*" ; Flags: ignoreversion recursesubdirs createallsubdirs
Source: E:\software\eo\repository\eo-ROOT\paradiseo-eo\src\utils\pipecom.cpp; DestDir: "{app}\paradiseo-eo\src\utils";  Excludes: "*.~*" ; Flags: ignoreversion recursesubdirs createallsubdirs
Source: E:\software\eo\repository\eo-ROOT\paradiseo-eo\src\eoCtrlCContinue.h; DestDir: "{app}\paradiseo-eo\src";  Excludes: "*.~*" ; Flags: ignoreversion recursesubdirs createallsubdirs
Source: E:\software\eo\repository\eo-ROOT\paradiseo-eo\src\eoCtrlCContinue.cpp; DestDir: "{app}\paradiseo-eo\src";  Excludes: "*.~*" ; Flags: ignoreversion recursesubdirs createallsubdirs

[Dirs]
Name: {app}\logs

[Code]
var
  GeneratorPage: TWizardPage;
  BuildModePage: TWizardPage;
  BuildProcessPage: TWizardPage;
  CMakeLookupPage: TWizardPage;
  ProgressPage: TOutputProgressWizardPage;
  GeneratorBox: TNewCheckListBox;
  BuildModeBox: TNewCheckListBox;
  SendReportBox: TCheckBox;
  Generator: String;
  CTestConfig: String;
  CMakeBinDir: String;
  ProgressBarLabel: TLabel;
  ProgressBar: TNewProgressBar;
  FolderTreeView: TFolderTreeView;
  CMakeAdditionalTags: String;
  OkToCopyLog : Boolean;
  TodaysName  : String;


function GetToday : String;
begin
  Result := GetDateTimeString ('yyyy/mm/dd', '-', #0);
end;

function GetTodaysName (Param: String): String;
begin
  if ('' = TodaysName) then
  begin
    TodaysName := GetToday ();
    end;
    Result := TodaysName;
end;

procedure SetCmakeGenerator();
begin
     if GeneratorBox.Checked[1] then
    begin
            Generator:='Visual Studio 9 2008' ;
            exit;
    end;
    if GeneratorBox.Checked[2] then
    begin
            Generator:='Visual Studio 9 2008 Win64' ;
            exit;
    end;
     if GeneratorBox.Checked[3] then
    begin
            Generator:='Visual Studio 8 2005' ;
            exit;
    end;
    if GeneratorBox.Checked[4] then
    begin
            Generator:='Visual Studio 8 2005 Win64' ;
            exit;
    end;
     if GeneratorBox.Checked[5] then
    begin
            Generator:='Visual Studio 7 .NET 2003' ;
            exit;
    end;
     if GeneratorBox.Checked[6] then
    begin
            Generator:='Visual Studio 7' ;
            exit;
    end;
     if GeneratorBox.Checked[7] then
    begin
            Generator:='Visual Studio 6' ;
            exit;
    end;
    if GeneratorBox.Checked[8] then
    begin
            Generator:='NMake Makefiles' ;
            exit;
    end;
    if GeneratorBox.Checked[9] then
    begin
            Generator:='MinGW Makefiles' ;
            exit;
    end;
    if GeneratorBox.Checked[10] then
    begin
            Generator:='Borland Makefiles' ;
            exit;
    end;
     if GeneratorBox.Checked[11] then
    begin
            Generator:='MSYS Makefiles' ;
            exit;
    end;
     if GeneratorBox.Checked[12] then
    begin
            Generator:='Watcom WMake'   ;
            exit;
    end;
end;


procedure SetCTestConfig();
var
  MinConfig: String;
begin
    MinConfig:=' -D ExperimentalStart -D ExperimentalBuild' ;

    if SendReportBox.Checked then
    begin
            CTestConfig:= MinConfig + ' -D ExperimentalSubmit' ;
    end;

     if BuildModeBox.Checked[1] then
    begin
            CMakeAdditionalTags:= ' -DENABLE_CMAKE_TESTING=FALSE';
            exit;
    end;
    if BuildModeBox.Checked[2] then
    begin
            CTestConfig:=CTestConfig + ' -D ExperimentalTest' ;
            CMakeAdditionalTags:= ' -DENABLE_CMAKE_TESTING=TRUE';
            exit;
    end;
end;

function isError(ErrorCode: Integer; PrintMsgBox: Boolean): Boolean;
begin
        if not (ErrorCode = 0) then
        begin
           if(ErrorCode = {#ctestMemCheckDummyError}) then begin
            Result:= false;
            exit;
          end;
           if(ErrorCode = {#dartSubmissionError}) then begin
            Result:= false;
            exit;
          end;
          if(PrintMsgBox) then begin
            MsgBox(CustomMessage('ErrorOccured') + ': [code='+ IntToStr(ErrorCode) + ']' + ' [' + SysErrorMessage(ErrorCode) + ']' , mbCriticalError, mb_Ok);
          end;
          Result:= true;
          
        end else begin
          Result:= false;
        end;
end;


function checkCMakeAvailable(Path: String): Integer;
var
  ErrorCode: Integer;
begin
   // launch CMake for MOEO
   ShellExec('open', Path + 'cmake.exe','','', SW_SHOWMINIMIZED, ewWaitUntilTerminated, ErrorCode);

   Result:=  ErrorCode;
end;



function LaunchEOBuildProcess():Integer;
var
  ErrorCode: Integer;
  FilePath: String;
begin
   Log('[LaunchEOBuildProcess] [begin]');
     
  // Need the app path
    FilePath := ExpandConstant('{app}');

   // launch CMake for EO
   Log('[LaunchEOBuildProcess] Launching: ' + CMakeBinDir + 'cmake.exe' + ' ..\' + ' -G"' + Generator + '"' + CMakeAdditionalTags);
   Log('[LaunchEOBuildProcess] From:  ' + FilePath +'\paradiseo-eo\build');
   ShellExec ('open',CMakeBinDir + 'cmake.exe',' ..\' + ' -G"' + Generator + '"' + CMakeAdditionalTags, FilePath +'\paradiseo-eo\build', SW_SHOWNORMAL, ewWaitUntilTerminated, ErrorCode);
   Log('[LaunchEOBuildProcess] Error code=' + IntToStr(ErrorCode));
   Log('[LaunchEOBuildProcess] [End]');
   Result:= ErrorCode;
end;


function LaunchEOCompilation():Integer;
var
  ErrorCode: Integer;
  FilePath: String;
begin
   Log('[LaunchEOCompilation] [begin]');

   // Need the app path
    FilePath := ExpandConstant('{app}');

   // launch CTest for EO
   Log('[LaunchEOCompilation] Launching: ' + CMakeBinDir + ' ctest.exe ' + CTestConfig);
   Log('[LaunchEOCompilation] From:  ' + FilePath +'\paradiseo-eo\build');
   Exec(ExpandConstant ('{sys}\CMD.EXE'), ' /C ' +  '"' + CMakeBinDir +  'ctest.exe' + ' "' + CTestConfig + ' > build-eo-' + GetTodaysName ('') + '.log',FilePath +'\paradiseo-eo\build', SW_SHOWNORMAL, ewWaitUntilTerminated,ErrorCode);
   FileCopy (FilePath +'\paradiseo-eo\build\build-eo-' + GetTodaysName ('') + '.log', ExpandConstant ('{app}\logs\build-eo-') + GetTodaysName ('') + '.log' , FALSE);
   Log('[LaunchEOCompilation] Error code=' + IntToStr(ErrorCode));
   Log('[LaunchEOCompilation] [End]');
   Result:= ErrorCode;
end;


function LaunchMOBuildProcess(): Integer;
var
  ErrorCode: Integer;
  FilePath: String;
begin
   Log('[LaunchMOBuildProcess] [begin]');

  // Need the app path
    FilePath := ExpandConstant('{app}');

   // launch CMake for MO
   Log('[LaunchMOBuildProcess] Launching: ' + CMakeBinDir + 'cmake.exe' + ' ..\' + ' -G"' + Generator + '" -Dconfig="'+FilePath + '\install.cmake"' + CMakeAdditionalTags);
   Log('[LaunchMOBuildProcess] From:  ' + FilePath +'\paradiseo-mo\build');
   ShellExec('open',CMakeBinDir+ 'cmake.exe ',' ..\' + ' -G"' + Generator + '" -Dconfig="'+FilePath + '\install.cmake"' + CMakeAdditionalTags,FilePath +'\paradiseo-mo\build', SW_SHOWNORMAL, ewWaitUntilTerminated, Errorcode);
   Log('[LaunchMOBuildProcess] Error code=' + IntToStr(ErrorCode));
   Log('[LaunchMOBuildProcess] [End]');
   Result:= ErrorCode;
end;


function LaunchMOCompilation():Integer;
var
  ErrorCode: Integer;
  FilePath: String;
begin
   Log('[LaunchMOCompilation] [begin]');

   // Need the app path
    FilePath := ExpandConstant('{app}');

   // launch CTest for MO
   Log('[LaunchMOCompilation] Launching: ' + CMakeBinDir + ' ctest.exe ' + CTestConfig);
   Log('[LaunchMOCompilation] From:  ' + FilePath +'\paradiseo-mo\build');
   Exec(ExpandConstant ('{sys}\CMD.EXE'), ' /C ' +  '"' + CMakeBinDir +  'ctest.exe' + ' "' + CTestConfig + ' > build-mo-' + GetTodaysName ('') + '.log',FilePath +'\paradiseo-mo\build', SW_SHOWNORMAL, ewWaitUntilTerminated,ErrorCode);
   FileCopy (FilePath +'\paradiseo-mo\build\build-mo-' + GetTodaysName ('') + '.log', ExpandConstant ('{app}\logs\build-mo-') + GetTodaysName ('') + '.log' , FALSE);
   Log('[LaunchMOCompilation] Error code=' + IntToStr(ErrorCode));
   Log('[LaunchMOCompilation] [End]');
   Result:= ErrorCode;
end;


function LaunchMOEOBuildProcess(): Integer;
var
  ErrorCode: Integer;
  FilePath: String;
begin
    // Need the app path
    FilePath := ExpandConstant('{app}');

   // launch CMake for MOEO
   Log('[LaunchMOEOBuildProcess] Launching: ' + CMakeBinDir + 'cmake.exe' + ' ..\' + ' -G"' + Generator + '" -Dconfig="'+FilePath + '\install.cmake"' + CMakeAdditionalTags);
   Log('[LaunchMOEOBuildProcess] From:  ' + FilePath +'\paradiseo-moeo\build');
   ShellExec('open', CMakeBinDir + 'cmake.exe', ' ..\' + ' -G"' + Generator + '"  -Dconfig="'+FilePath + '\install.cmake"' + CMakeAdditionalTags, FilePath +'\paradiseo-moeo\build', SW_SHOWNORMAL, ewWaitUntilTerminated, ErrorCode);
   Log('[LaunchMOEOBuildProcess] Error code=' + IntToStr(ErrorCode));
   Log('[LaunchMOEOBuildProcess] [End]');
   Result:= ErrorCode;
   Result:=  ErrorCode;
end;


function LaunchMOEOCompilation():Integer;
var
  ErrorCode: Integer;
  FilePath: String;
begin
   Log('[LaunchMOEOCompilation] [begin]');

   // Need the app path
    FilePath := ExpandConstant('{app}');

   // launch CTest for MOEO
   Log('[LaunchMOEOCompilation] Launching: ' + CMakeBinDir + ' ctest.exe ' + CTestConfig);
   Log('[LaunchMOEOCompilation] From:  ' + FilePath +'\paradiseo-moeo\build');
   Exec(ExpandConstant ('{sys}\CMD.EXE'), ' /C ' +  '"' + CMakeBinDir +  'ctest.exe' + ' "' + CTestConfig + ' > build-moeo-' + GetTodaysName ('') + '.log',FilePath +'\paradiseo-moeo\build', SW_SHOWNORMAL, ewWaitUntilTerminated,ErrorCode);
   FileCopy (FilePath +'\paradiseo-moeo\build\build-moeo-' + GetTodaysName ('') + '.log', ExpandConstant ('{app}\logs\build-moeo-') + GetTodaysName ('') + '.log' , FALSE);
   Log('[LaunchMOEOCompilation] Error code=' + IntToStr(ErrorCode));
   Log('[LaunchMOEOCompilation] [End]');
   Result:= ErrorCode;
end;


procedure updateProgressBar(StartFrom: Integer;ProgressPurcentage: Integer);
var
      I: Integer;
begin
      try
         for I := StartFrom to ProgressPurcentage do begin
         ProgressPage.SetProgress(I, 100);
         Sleep(50);
         end;
      finally
      end;
end;


function NextButtonClick(CurPageID: Integer): Boolean;
begin

  if (CurPageID = BuildProcessPage.ID)  then
  begin
       ProgressPage.SetText('',CustomMessage('LaunchingBuildProcess'));
       updateProgressBar(0,0);
       ProgressPage.Show;

       SetCmakeGenerator();
       SetCTestConfig();

       updateProgressBar(0,5);
       //***************** EO *************************
         ProgressPage.SetText('',CustomMessage('LaunchingEOBuildProcess'));
         if (isError(launchEOBuildProcess(),true)) then
          begin
           ProgressPage.SetText(CustomMessage('Error'), CustomMessage('ErrorAbort'));
           MsgBox(CustomMessage('CannotCompleteInstall')+' ParadisEO-EO' , mbCriticalError, mb_Ok);
           ProgressPage.Hide;
           Result := True;
           exit;
         end;
       updateProgressBar(5,20);

        ProgressPage.SetText('',CustomMessage('LaunchingEOCompilation'));
       if (isError(LaunchEOCompilation(),true)) then
          begin
           ProgressPage.SetText(CustomMessage('Error'), CustomMessage('ErrorAbort'));
           MsgBox(CustomMessage('CannotCompleteInstall')+' ParadisEO-EO' , mbCriticalError, mb_Ok);
           ProgressPage.Hide;
           Result := True;
           exit;
         end;
       updateProgressBar(20,50);
       
       
        //***************** MO *************************
        if IsComponentSelected('mo') then
         begin
          ProgressPage.SetText('',CustomMessage('LaunchingMOBuildProcess'));
          if (isError(launchMOBuildProcess(),true)) then
           begin
           ProgressPage.SetText(CustomMessage('Error'), CustomMessage('ErrorAbort'));
             MsgBox(CustomMessage('CannotCompleteInstall')+' ParadisEO-MO' , mbCriticalError, mb_Ok);
            ProgressPage.Hide;
            Result := True;
            exit;
          end;
          updateProgressBar(50,60);

           ProgressPage.SetText('',CustomMessage('LaunchingMOCompilation'));
           if (isError(LaunchMOCompilation(),true)) then
            begin
             ProgressPage.SetText(CustomMessage('Error'), CustomMessage('ErrorAbort'));
             MsgBox(CustomMessage('CannotCompleteInstall')+' ParadisEO-MO' , mbCriticalError, mb_Ok);
             ProgressPage.Hide;
             Result := True;
             exit;
           end;
           updateProgressBar(60,80);
         end else
            updateProgressBar(50,60);
            
      //***************** MOEO *************************
        if IsComponentSelected('moeo') then
         begin
          ProgressPage.SetText('',CustomMessage('LaunchingMOEOBuildProcess'));
          if (isError(launchMOEOBuildProcess(),true)) then
           begin
           ProgressPage.SetText(CustomMessage('Error'), CustomMessage('ErrorAbort'));
             MsgBox(CustomMessage('CannotCompleteInstall')+' ParadisEO-MOEO' , mbCriticalError, mb_Ok);
            ProgressPage.Hide;
            Result := True;
            exit;
          end;
          updateProgressBar(60,75);

           ProgressPage.SetText('',CustomMessage('LaunchingMOEOCompilation'));
           if (isError(LaunchMOEOCompilation(),true)) then
            begin
             ProgressPage.SetText(CustomMessage('Error'), CustomMessage('ErrorAbort'));
             MsgBox(CustomMessage('CannotCompleteInstall')+' ParadisEO-MOEO' , mbCriticalError, mb_Ok);
             ProgressPage.Hide;
             Result := True;
             exit;
           end;
           updateProgressBar(75,100);
         end else
            updateProgressBar(60,100);
            
      ProgressPage.SetText(CustomMessage('BPFinished'), CustomMessage('BPSuccessfull'));
      sleep(2000);
      ProgressPage.Hide;
  end;
   Result := True;
end;


function Skeleton_NextButtonClick(Page: TWizardPage): Boolean;
begin
    { Get the Cmake directory provided by the user }
     CMakeBinDir:= '"' + FolderTreeView.Directory + '\';
     if isError(checkCMakeAvailable(CMakeBinDir),false) then  begin
          CMakeBinDir:= FolderTreeView.Directory + '\' + 'bin\';
          if isError(checkCMakeAvailable(CMakeBinDir),false) then  begin
             MsgBox(CustomMessage('CMakeNotFound'), mbCriticalError, mb_Ok);
             Result := False;
             exit;
         end else
             Result := True;   exit;
        
        MsgBox(CustomMessage('CMakeNotFound'), mbCriticalError, mb_Ok);
        Result := False;
     end else
        Result := True;
end;


Procedure CMakeURLLabelOnClick(Sender: TObject);
var
  ErrorCode: Integer;
begin
  ShellExec('open', 'http://www.cmake.org/HTML/Download.html', '', '', SW_SHOWNORMAL, ewNoWait, ErrorCode);
end;



procedure CreateTheWizardPages;
var
   Lbl1,Lbl2,Lbl3,Lbl4,Lbl5,Lbl6: TLabel;
   CMakeURLLabel: TNewStaticText;
begin

  if (isError(checkCMakeAvailable(''),False))   then begin
       CMakeLookupPage := CreateCustomPage({#cmakeLookupWizardPageIndex},CustomMessage('PathToCMakeTitle'),CustomMessage('PathToCMakeSubtitle'));
       FolderTreeView := TFolderTreeView.Create(CMakeLookupPage);
       FolderTreeView.Top := ScaleY(40)
       FolderTreeView.Width := CMakeLookupPage.SurfaceWidth;
       FolderTreeView.Height := CMakeLookupPage.SurfaceHeight;
       FolderTreeView.Parent := CMakeLookupPage.Surface;
       FolderTreeView.Directory := '';

       Lbl4 := TLabel.Create(CMakeLookupPage);
       Lbl4.Top := ScaleY(20);
       Lbl4.Caption := CustomMessage('CMakeDownloadMsg');
       Lbl4.AutoSize := True;
       Lbl4.Parent := CMakeLookupPage.Surface;
       Lbl4.Font.Size := 8 ;
       Lbl4.Top := ScaleY(0);
       Lbl4.Left := ScaleX(5);
       
       CMakeURLLabel := TNewStaticText.Create(CMakeLookupPage);
       CMakeURLLabel.Caption := 'http://www.cmake.org/HTML/Download.html';
       CMakeURLLabel.Cursor := crHand;
       CMakeURLLabel.OnClick:= @CMakeURLLabelOnClick;
       CMakeURLLabel.Parent := CMakeLookupPage.Surface;
       CMakeURLLabel.Font.Style := CMakeURLLabel.Font.Style + [fsUnderline];
       CMakeURLLabel.Font.Color := clBlue;
       CMakeURLLabel.Top := ScaleY(0);
       CMakeURLLabel.Left := ScaleX(170);
       
       CMakeLookupPage.OnNextButtonClick := @Skeleton_NextButtonClick;
  end;
  
  GeneratorPage := CreateCustomPage({#generatorWizardPageIndex}, CustomMessage('ChooseGeneratorTitle'), CustomMessage('ChooseGeneratorSubtitle'));
  GeneratorBox := TNewCheckListBox.Create(GeneratorPage);
  GeneratorBox.Top :=  ScaleY(0);
  GeneratorBox.Width := GeneratorPage.SurfaceWidth;
  GeneratorBox.Height := ScaleY(250);
  GeneratorBox.BorderStyle := bsNone;
  GeneratorBox.ParentColor := True;
  GeneratorBox.MinItemHeight := WizardForm.TasksList.MinItemHeight;
  GeneratorBox.ShowLines := False;
  GeneratorBox.WantTabs := True;
  GeneratorBox.Parent := GeneratorPage.Surface;
  GeneratorBox.AddGroup(CustomMessage('SelectCompiler'), '', 0, nil);
  GeneratorBox.AddRadioButton('Visual Studio 9 2008', '', 0, True, True, nil);
  GeneratorBox.AddRadioButton('Visual Studio 9 2008 Win64', '', 0, False, True, nil);
  GeneratorBox.AddRadioButton('Visual Studio 8 2005', '', 0, False, True, nil);
  GeneratorBox.AddRadioButton('Visual Studio 8 2005 Win64', '', 0, False, True, nil);
  GeneratorBox.AddRadioButton('Visual Studio 7 .NET 2003', '', 0, False, True, nil);
  GeneratorBox.AddRadioButton('Visual Studio 7', '', 0, False, True, nil);
  GeneratorBox.AddRadioButton('Visual Studio 6', '', 0, False, True, nil);
  GeneratorBox.AddRadioButton('NMake', '', 0, False, True, nil);
  GeneratorBox.AddRadioButton('MinGW', '', 0, False, True, nil);
  GeneratorBox.AddRadioButton('Borland', '', 0, False, True, nil);
  GeneratorBox.AddRadioButton('MSYS', '', 0, False, True, nil);
  GeneratorBox.AddRadioButton('Watcom WMake', '', 0, False, True, nil);


  BuildProcessPage := CreateCustomPage({#launchBuildWizardPageIndex},CustomMessage('GenCMakeFiles'),CustomMessage('BuildProcess'));

  BuildModePage := CreateCustomPage({#buildModeWizardPageIndex}, CustomMessage('BuildMode'), '');
  BuildModeBox := TNewCheckListBox.Create(BuildModePage);
  BuildModeBox.Top :=  ScaleY(0);
  BuildModeBox.Width := BuildModePage.SurfaceWidth;
  BuildModeBox.Height := ScaleY(80);
  BuildModeBox.BorderStyle := bsNone;
  BuildModeBox.ParentColor := True;
  BuildModeBox.MinItemHeight := WizardForm.TasksList.MinItemHeight;
  BuildModeBox.ShowLines := False;
  BuildModeBox.WantTabs := True;
  BuildModeBox.Parent := BuildModePage.Surface;
  BuildModeBox.AddGroup(CustomMessage('SelectBuildMode'), '', 0, nil);
  BuildModeBox.AddRadioButton('Normal = Release ' + CustomMessage('Recommended'), '', 0, True, True, nil);
  BuildModeBox.AddRadioButton('Debug', '', 0, False, True, nil);

  SendReportBox := TCheckBox.Create(BuildModePage);
  SendReportBox.Top :=  BuildModeBox.Top + BuildModeBox.Height + ScaleY(90);
  SendReportBox.Width := BuildModePage.SurfaceWidth;
  SendReportBox.Height := ScaleY(15);
  SendReportBox.Caption := CustomMessage('AcceptSendReport');
  SendReportBox.Checked := True;
  SendReportBox.Parent := BuildModePage.Surface;
  SendReportBox.Font.Size := 7;
  
  Lbl1 := TLabel.Create(BuildModePage);
  Lbl1.Top := SendReportBox.Top + SendReportBox.Height + ScaleY(5);
  Lbl1.Caption := CustomMessage('NoInfoSend1');
  Lbl1.AutoSize := True;
  Lbl1.Parent := BuildModePage.Surface;
  Lbl1.Font.Size := 7 ;
  Lbl1.Left := 15;
  
  Lbl2 := TLabel.Create(BuildModePage);
  Lbl2.Top := Lbl1.Top + Lbl1.Height + ScaleY(5);
  Lbl2.Caption := CustomMessage('NoInfoSend2');
  Lbl2.AutoSize := True;
  Lbl2.Parent := BuildModePage.Surface;
  Lbl2.Font.Size := 7 ;
  Lbl2.Left := 15;
  
  Lbl3 := TLabel.Create(BuildProcessPage);
  Lbl3.Top := ScaleY(20);
  Lbl3.Caption :=CustomMessage('NextGenCaption');
  Lbl3.AutoSize := True;
  Lbl3.Parent := BuildProcessPage.Surface;

  Lbl5 := TLabel.Create(BuildProcessPage);
  Lbl5.Top := ScaleY(40);
  Lbl5.Left := ScaleX(-3);
  Lbl5.Caption :=CustomMessage('NextGenCaptionPgmBegin');
  Lbl5.AutoSize := True;
  Lbl5.Parent := BuildProcessPage.Surface;
  
  ProgressPage := CreateOutputProgressPage(CustomMessage('ProcessingCMake'),CustomMessage('BuildProcess'));
end;

procedure AboutButtonOnClick(Sender: TObject);
begin
  MsgBox(CustomMessage('DolphinMsg'), mbInformation, mb_Ok);
end;


procedure URLLabelOnClick(Sender: TObject);
var
  ErrorCode: Integer;
begin
  ShellExec('open', 'http://paradiseo.gforge.inria.fr', '', '', SW_SHOWNORMAL, ewNoWait, ErrorCode);
end;


procedure InitializeWizard();
var
  AboutButton, CancelButton: TButton;
  URLLabel: TNewStaticText;
begin
  CreateTheWizardPages;
  CancelButton := WizardForm.CancelButton;

  AboutButton := TButton.Create(WizardForm);
  AboutButton.Left := WizardForm.ClientWidth - CancelButton.Left - CancelButton.Width - ScaleX(5);
  AboutButton.Top := CancelButton.Top;
  AboutButton.Width := CancelButton.Width;
  AboutButton.Height := CancelButton.Height;
  AboutButton.Caption := '&About...';
  AboutButton.OnClick := @AboutButtonOnClick;
  AboutButton.Parent := WizardForm;

  URLLabel := TNewStaticText.Create(WizardForm);
  URLLabel.Caption := 'http://paradiseo.gforge.inria.fr';
  URLLabel.Cursor := crHand;
  URLLabel.OnClick := @URLLabelOnClick;
  URLLabel.Parent := WizardForm;
  URLLabel.Font.Style := URLLabel.Font.Style + [fsUnderline];
  URLLabel.Font.Color := clBlue;
  URLLabel.Top := AboutButton.Top + AboutButton.Height - URLLabel.Height - 2;
  URLLabel.Left := AboutButton.Left + AboutButton.Width + ScaleX(10);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssDone then
    OkToCopyLog := True;
end;


procedure DeinitializeSetup();
begin
  if OkToCopyLog then
    FileCopy (ExpandConstant ('{log}'), ExpandConstant ('{app}\logs\install-') + GetTodaysName ('') + '.log' , FALSE);
  RestartReplace (ExpandConstant ('{log}'), '');
end;


[UninstallDelete]
Type: files; Name: "{app}\*"
Type: filesandordirs; Name: "{app}\*"