"""Package the native x86_64 onedir bundle as both deb and rpm."""
import shutil
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
VERSION='0.9.0'

def main():
    stage=ROOT/'build/linux-package'
    if stage.exists():shutil.rmtree(stage)
    (stage/'opt').mkdir(parents=True)
    shutil.copytree(ROOT/'dist/cc-media-studio',stage/'opt/cc-media-studio')
    binary=stage/'usr/bin/cc-media-studio';binary.parent.mkdir(parents=True)
    cli=stage/'usr/bin/cc-media-cli'
    cli.write_text('#!/bin/sh\nexec /opt/cc-media-studio/cc-media-cli "$@"\n');cli.chmod(0o755)
    binary.write_text('#!/bin/sh\nexec /opt/cc-media-studio/cc-media-studio --browser "$@"\n');binary.chmod(0o755)
    for src,dest in [('packaging/cc-media-studio.desktop','usr/share/applications/cc-media-studio.desktop'),
                     ('packaging/icon.svg','usr/share/icons/hicolor/scalable/apps/cc-media-studio.svg'),
                     ('LICENSE','usr/share/licenses/cc-media-studio/LICENSE'),('NOTICE','usr/share/licenses/cc-media-studio/NOTICE')]:
        target=stage/dest;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/src,target)
    output=ROOT/'release';output.mkdir(exist_ok=True)
    control=stage/'DEBIAN/control';control.parent.mkdir()
    control.write_text(f'''Package: cc-media-studio
Version: {VERSION}
Architecture: amd64
Maintainer: CC <noreply@users.noreply.github.com>
Depends: libc6 (>= 2.35), libstdc++6
Section: graphics
Priority: optional
Homepage: https://github.com/cc6pzupa/free-media-studio
Description: Image and video creation with Kaggle GPU compute
 Includes the Python client runtime. A desktop web browser is required.
''')
    subprocess.run(['dpkg-deb','--root-owner-group','--build',str(stage),str(output/f'CC-Media-Studio-{VERSION}-linux-amd64.deb')],check=True)
    shutil.rmtree(control.parent)
    top=ROOT/'build/rpm'
    for folder in ('BUILD','BUILDROOT','RPMS','SOURCES','SPECS','SRPMS'):(top/folder).mkdir(parents=True,exist_ok=True)
    spec=top/'SPECS/cc-media-studio.spec'
    spec.write_text(f'''Name: cc-media-studio
Version: {VERSION}
Release: 1
Summary: Image and video creation with Kaggle GPU compute
License: LicenseRef-PolyForm-Noncommercial-1.0.0
URL: https://github.com/cc6pzupa/free-media-studio
BuildArch: x86_64
AutoReqProv: no
Requires: glibc >= 2.35
Requires: libstdc++
%description
Includes the Python client runtime. A desktop web browser is required.
%install
mkdir -p %{{buildroot}}
cp -a "{stage}/." %{{buildroot}}/
%files
/opt/cc-media-studio
/usr/bin/cc-media-studio
/usr/bin/cc-media-cli
/usr/share/applications/cc-media-studio.desktop
/usr/share/icons/hicolor/scalable/apps/cc-media-studio.svg
/usr/share/licenses/cc-media-studio
''')
    subprocess.run(['rpmbuild','--define',f'_topdir {top}','--define','__os_install_post %{nil}','-bb',str(spec)],check=True)
    rpm=next((top/'RPMS/x86_64').glob('*.rpm'));shutil.copy2(rpm,output/f'CC-Media-Studio-{VERSION}-linux-x86_64.rpm')

if __name__=='__main__':main()
