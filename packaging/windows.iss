#ifndef AppVersion
  #define AppVersion "0.9.0"
#endif
[Setup]
AppId={{A34134EA-26E7-41E8-BB9E-B0C2C6760391}
AppName=CC Media Studio
AppVersion={#AppVersion}
AppPublisher=CC
AppPublisherURL=https://github.com/cc6pzupa/free-media-studio
DefaultDirName={localappdata}\Programs\CC Media Studio
DefaultGroupName=CC Media Studio
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\release
OutputBaseFilename=CC-Media-Studio-{#AppVersion}-windows-x64-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no
LicenseFile=..\LICENSE
UninstallDisplayIcon={app}\cc-media-studio.exe
[Tasks]
Name: desktopicon; Description: "Create a desktop shortcut"; Flags: unchecked
[Files]
Source: "..\dist\cc-media-studio\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
[Icons]
Name: "{group}\CC Media Studio"; Filename: "{app}\cc-media-studio.exe"
Name: "{autodesktop}\CC Media Studio"; Filename: "{app}\cc-media-studio.exe"; Tasks: desktopicon
[Run]
Filename: "{app}\cc-media-studio.exe"; Description: "Open CC Media Studio"; Flags: nowait postinstall skipifsilent
