; ParadisEO install script
; Author: Thomas Legrand

#define generatorWizardPageIndex= 7
#define launchBuildWizardPageIndex= 12

[Setup]
AppName=ParadisEO
AppVerName=ParadisEO-ix86-1.0
AppPublisher=INRIA Futurs Dolphin Project-team
AppPublisherURL=http://paradiseo.gforge.inria.fr
AppSupportURL=http://paradiseo.gforge.inria.fr
AppUpdatesURL=http://paradiseo.gforge.inria.fr
DefaultDirName={pf}\ParadisEO
DefaultGroupName=ParadisEO
LicenseFile=E:\software\paradisEO\repository\trunk\LICENSE
OutputDir=E:\software\paradisEO\windows installer\compiler output
OutputBaseFilename=paradiseo-1.0-win32-preinstaller
Compression=lzma/max
SolidCompression=yes
WizardImageFile=E:\software\paradisEO\repository\utilities\trunk\windows\img\paradiseo.bmp
SetupIconFile=E:\software\paradisEO\repository\utilities\trunk\windows\img\paradiseo.ico
UninstallDisplayName=ParadisEO
WindowVisible=False
RestartIfNeededByRun=False
ShowTasksTreeLines=True
VersionInfoVersion=1.0
VersionInfoCompany=INRIA
VersionInfoDescription=ParadisEO
VersionInfoTextVersion=ParadisEO

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
english.LaunchingBuildProcess=Launching CMake build process...
english.error=Error
english.ErrorAbort=Error,abort.
english.CannotCompleteInstall=Impossible to complete the install of
english.BPFinished=Finished
english.BPSuccessfull=The build process has been successfully performed.
english.SelectCompiler=Select the program you want to use to compile:
english.ChooseGenerator=ParadisEO can be compiled using several generators.
english.GenCMakeFiles=Generate CMake configuration files
english.BuildProcess=
english.NextGenCaption=Click on the 'Next' button to launch the build process and generate the configuration files
english.ProcessingCMake=Generating the configuration files...
english.DolphinMsg=ParadisEO: An INRIA Dolphin Team project - 2007

french.CMakeMissing=CMake n'a pas été détecté sur votre ordinateur. CMake doit être installé pour utiliser ParadisEO.
french.FullInstall=Installation complète
french.CustomInstall=Installation personnalisée
french.EoDescription= EO: Evolving Objects: Librairie dédiée aux méthodes évolutionnaires
french.MoDescription= MO: Moving Objects: Métaheuristiques à base de solutions uniques
french.MoeoDescription= MOEO: Multi Objective Evolving Objects: Module multi-objectif
french.ErrorOccured=Une erreur est survenue
french.LaunchingBuildProcess=Construction des fichiers de configuration (build process)...
french.error=Erreur
french.ErrorAbort=Une erreur est survenue, installation annulée.
french.CannotCompleteInstall=Impossible de terminer l'installation de
french.BPFinished=Fin de la construction des fichiers de configuration
french.BPSuccessfull=Succès.
french.SelectCompiler=Sélectionnez le programme que vous souhaitez utiliser pour compiler:
french.ChooseGenerator=ParadisEO peut être compiler par plusieurs programmes.
french.GenCMakeFiles=Générer les fichiers de configuration CMake
french.BuildProcess=
french.NextGenCaption=Cliquez sur le bouton 'Suivant' pour lancer CMake et générer les fichiers de configuration
french.ProcessingCMake=Génération des fichiers de configuration en cours...
french.DolphinMsg=ParadisEO: Un projet de l'équipe INRIA Dolphin - 2007

[Types]
Name: "custom"; Description: {cm:CustomInstall}; Flags: iscustom
Name: "full"; Description: {cm:FullInstall}

[Components]
Name: eo; Description: {cm:EoDescription}; Types: full custom; Flags: fixed
Name: mo; Description:{cm:MoDescription}; Types: full custom;
Name: moeo; Description: {cm:MoeoDescription}; Types: full custom;

[Files]
Source: "E:\software\paradisEO\repository\tags\paradiseo-ix86-1.0\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs


[Code]
var
  GeneratorPage: TWizardPage;
  BuildProcessPage: TWizardPage;
  ProgressPage: TOutputProgressWizardPage;
  GeneratorBox: TNewCheckListBox;
  Generator: String;
  ProgressBarLabel: TLabel;
  ProgressBar: TNewProgressBar;

  
procedure SetCmakeGenerator();
begin
     if GeneratorBox.Checked[1] then
    begin
            Generator:='Visual Studio 8 2005' ;
            exit;
    end;
     if GeneratorBox.Checked[2] then
    begin
            Generator:='Visual Studio 7 .NET 2003' ;
            exit;
    end;
     if GeneratorBox.Checked[3] then
    begin
            Generator:='NMake'       ;
            exit;
    end;
     if GeneratorBox.Checked[4] then
    begin
            Generator:='MinGW' ;
            exit;
    end;
     if GeneratorBox.Checked[5] then
    begin
            Generator:='Borland'  ;
            exit;
    end;
     if GeneratorBox.Checked[6] then
    begin
            Generator:='MSYS'   ;
            exit;
    end;
     if GeneratorBox.Checked[7] then
    begin
            Generator:='WMake'   ;
            exit;
    end;
end;


function isError(ErrorCode: Integer): Boolean;
begin
        if not (ErrorCode = 0) then
        begin
          MsgBox(CustomMessage('ErrorOccured') + ': [code='+ IntToStr(ErrorCode) + ']' , mbCriticalError, mb_Ok);
          Result:= true;
        end else begin
          Result:= false;
        end;
end;


function checkCMakeAvailable(): Integer;
var
  ErrorCode: Integer;
  FilePath: String;
begin
    // Need the app path
    FilePath := ExpandConstant('{app}');

   // launch CMake for MOEO
   ShellExec('open', 'cmake.exe','','', SW_SHOWNORMAL, ewWaitUntilTerminated, ErrorCode);

   Result:=  ErrorCode;
end;

function LaunchEOBuildProcess():Integer;
var
  ErrorCode: Integer;
  FilePath: String;
begin
  // Need the app path
    FilePath := ExpandConstant('{app}');

   // launch CMake for EO
   ShellExec('open', 'cmake.exe', ' ..\' + ' -G"' + Generator + '"', FilePath +'\paradiseo-eo\build', SW_SHOWNORMAL, ewWaitUntilTerminated, ErrorCode);

   Result:=  ErrorCode;
end;


function LaunchMOBuildProcess(): Integer;
var
  ErrorCode: Integer;
  FilePath: String;
begin
  // Need the app path
    FilePath := ExpandConstant('{app}');

   // launch CMake for MOEO
   ShellExec('open', 'cmake.exe', ' ..\' + ' -G"' + Generator + '"  -Dconfig="'+FilePath + '\install.cmake"', FilePath +'\paradiseo-mo\build', SW_SHOWNORMAL, ewWaitUntilTerminated, ErrorCode);

   Result:=  ErrorCode;
end;


function LaunchMOEOBuildProcess(): Integer;
var
  ErrorCode: Integer;
  FilePath: String;
begin
    // Need the app path
    FilePath := ExpandConstant('{app}');

   // launch CMake for MOEO
   ShellExec('open', 'cmake.exe', ' ..\' + ' -G"' + Generator + '"  -Dconfig="'+FilePath + '\install.cmake"', FilePath +'\paradiseo-moeo\build', SW_SHOWNORMAL, ewWaitUntilTerminated, ErrorCode);

   Result:=  ErrorCode;
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


procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssInstall then begin
     if (isError(checkCMakeAvailable()))   then begin
          MsgBox(CustomMessage('CMakeMissing'), mbCriticalError, mb_Ok);
          Abort;
    end;
  end;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  I: Integer;
begin
  if (CurPageID = BuildProcessPage.ID)  then begin
       ProgressPage.SetText('',CustomMessage('LaunchingBuildProcess'));
       updateProgressBar(0,0);
       ProgressPage.Show;

       SetCmakeGenerator();
       updateProgressBar(0,5);

         if (isError(launchEOBuildProcess())) then
          begin
           ProgressPage.SetText(CustomMessage('Error'), CustomMessage('ErrorAbort'));
           MsgBox(CustomMessage('CannotCompleteInstall')+'ParadisEO-EO' , mbCriticalError, mb_Ok);
           ProgressPage.Hide;
           Result := True;
           exit;
         end;
       updateProgressBar(5,50);

        if IsComponentSelected('mo') then
         begin
         if (isError(launchMOBuildProcess())) then
          begin
          ProgressPage.SetText(CustomMessage('Error'), CustomMessage('ErrorAbort'));
            MsgBox(CustomMessage('CannotCompleteInstall')+'ParadisEO-MO' , mbCriticalError, mb_Ok);
           ProgressPage.Hide;
           Result := True;
           exit;
         end;
        end;
       updateProgressBar(50,80);

       if IsComponentSelected('moeo') then
        begin
        if (isError(launchMOEOBuildProcess())) then
         begin
         ProgressPage.SetText(CustomMessage('Error'), CustomMessage('ErrorAbort'));
           MsgBox(CustomMessage('CannotCompleteInstall')+'ParadisEO-MOEO' , mbCriticalError, mb_Ok);
          ProgressPage.Hide;
          Result := True;
          exit;
        end;
       end;
       updateProgressBar(80,100);

       ProgressPage.SetText(CustomMessage('BPFinished'), CustomMessage('BPSuccessfull'));
       sleep(2000);
       ProgressPage.SetText('Fin','Vous devez maintenant compiler');
       ProgressPage.Hide;
  end;
   Result := True;
end;




procedure CreateTheWizardPages;
var
   Lbl: TLabel;
begin
  GeneratorPage := CreateCustomPage({#generatorWizardPageIndex}, CustomMessage('ChooseGenerator'), '');
  GeneratorBox := TNewCheckListBox.Create(GeneratorPage);
  GeneratorBox.Top :=  ScaleY(0);
  GeneratorBox.Width := GeneratorPage.SurfaceWidth;
  GeneratorBox.Height := ScaleY(180);
  GeneratorBox.BorderStyle := bsNone;
  GeneratorBox.ParentColor := True;
  GeneratorBox.MinItemHeight := WizardForm.TasksList.MinItemHeight;
  GeneratorBox.ShowLines := False;
  GeneratorBox.WantTabs := True;
  GeneratorBox.Parent := GeneratorPage.Surface;
  GeneratorBox.AddGroup(CustomMessage('SelectCompiler'), '', 0, nil);
  GeneratorBox.AddRadioButton('Visual Studio 8 2005', '', 0, True, True, nil);
  GeneratorBox.AddRadioButton('Visual Studio 7 .NET 2003', '', 0, False, True, nil);
  GeneratorBox.AddRadioButton('NMake', '', 0, False, True, nil);
  GeneratorBox.AddRadioButton('MinGW', '', 0, False, True, nil);
  GeneratorBox.AddRadioButton('Borland', '', 0, False, True, nil);
  GeneratorBox.AddRadioButton('MSYS', '', 0, False, True, nil);
  GeneratorBox.AddRadioButton('WMake', '', 0, False, True, nil);
  
  BuildProcessPage := CreateCustomPage({#launchBuildWizardPageIndex},CustomMessage('GenCMakeFiles'),CustomMessage('BuildProcess'));

  Lbl := TLabel.Create(BuildProcessPage);
  Lbl.Top := ScaleY(20);
  Lbl.Caption :=CustomMessage('NextGenCaption');
  Lbl.AutoSize := True;
  Lbl.Parent := BuildProcessPage.Surface;

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
  BackgroundBitmapImage: TBitmapImage;
  BackgroundBitmapText: TNewStaticText;
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


[UninstallDelete]
Type: files; Name: "{app}\*"
Type: filesandordirs; Name: "{app}\*"


