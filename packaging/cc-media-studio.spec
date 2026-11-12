# Build on the target OS/CPU: python -m PyInstaller packaging/cc-media-studio.spec
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules
root=Path(SPECPATH).parent
# Remote sources are uploaded verbatim to Kaggle. Include them as data as well
# as compiled client modules; the client never installs torch or diffusers.
datas=[(str(p),'.') for pattern in ('*.py','*.html','*.ipynb','requirements*.txt') for p in root.glob(pattern)]
datas += [(str(root/name),'.') for name in ('LICENSE','NOTICE','COMMERCIAL.md')]
hidden=['studio_worker','studio_core','studio_client','account_backend','session_backend','batch_backend',
        'video_session','video_registry','model_registry','easy_gateway','desktop','mcp_server']
hidden+=collect_submodules('mcp', filter=lambda name: not name.startswith('mcp.cli'))
if sys.platform in ('win32','darwin'):hidden+=collect_submodules('webview')
a=Analysis([str(root/'app_entry.py')],pathex=[str(root)],binaries=[],datas=datas,hiddenimports=hidden,
           hookspath=[],runtime_hooks=[],excludes=['torch','diffusers','transformers','tkinter'],noarchive=False)
pyz=PYZ(a.pure)
exe=EXE(pyz,a.scripts,[],exclude_binaries=True,name='cc-media-studio',debug=False,strip=False,
        upx=False,console=sys.platform not in ('win32','darwin'))
cli=EXE(pyz,a.scripts,[],exclude_binaries=True,name='cc-media-cli',debug=False,strip=False,upx=False,console=True)
coll=COLLECT(exe,cli,a.binaries,a.datas,strip=False,upx=False,name='cc-media-studio')
if sys.platform=='darwin':
    app=BUNDLE(coll,name='CC Media Studio.app',bundle_identifier='studio.ccmedia.desktop',
        info_plist={'CFBundleExecutable':'cc-media-studio','CFBundleShortVersionString':'0.9.0','CFBundleVersion':'0.9.0',
                    'NSHighResolutionCapable':True,'NSRequiresAquaSystemAppearance':False})
