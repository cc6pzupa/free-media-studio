#!/usr/bin/env bash
# Installs the APK on the CI emulator and checks that the in-app Python
# worker and GUI server come up and answer through the WebView's address.
set -euo pipefail
apk=$1
out=${2:-android-smoke}
mkdir -p "$out"
adb install -r "$apk"
adb shell pm grant com.ccmedia.studio android.permission.POST_NOTIFICATIONS || true
adb logcat -c || true
adb shell am start -W -n com.ccmedia.studio/.MainActivity
adb forward tcp:8787 tcp:8787
ok=0
for _ in $(seq 1 120); do
  if curl -fsS --max-time 3 http://127.0.0.1:8787/health >/dev/null 2>&1; then ok=1; break; fi
  sleep 2
done
collect() {
  adb exec-out screencap -p > "$out/screen.png" || true
  adb logcat -d > "$out/logcat.txt" || true
}
if [ "$ok" != 1 ]; then collect; echo "Studio server did not start"; grep -iE "python|chaquopy|ccmedia|AndroidRuntime" "$out/logcat.txt" | tail -80; exit 1; fi
# Save each response before searching it: "curl | grep -q" fails under
# pipefail when grep exits early and curl cannot finish writing.
fetch() { curl -fsS --max-time 20 "$@"; }
fetch -o "$out/health.json" http://127.0.0.1:8787/health
grep -q '"qwen-studio"' "$out/health.json"
fetch -o "$out/index.html" http://127.0.0.1:8787/
grep -q 'CC Media Studio' "$out/index.html"
nonce=$(grep -o "const nonce='[^']*'" "$out/index.html" | cut -d"'" -f2)
# Goes through StudioClient to the worker thread inside the app.
fetch -o "$out/configuration.json" -H "X-Local-Session: $nonce" http://127.0.0.1:8787/configuration
grep -q '"configured"' "$out/configuration.json"
for page in video gallery; do
  fetch -o "$out/$page.html" "http://127.0.0.1:8787/$page"
  grep -q 'CC Media Studio' "$out/$page.html"
done
# The page itself must have loaded in the WebView (not the "starting" text).
sleep 8
adb shell dumpsys activity activities > "$out/activities.txt"
grep -q "com.ccmedia.studio/.MainActivity" "$out/activities.txt"
collect
if grep -q "FATAL EXCEPTION" "$out/logcat.txt"; then grep -A20 "FATAL EXCEPTION" "$out/logcat.txt"; exit 1; fi
# Screenshots of each page for the guide, printed into the log as small JPEGs.
for page in image chat video gallery; do
  adb shell am start -n com.ccmedia.studio/.MainActivity --es page "$page" >/dev/null
  sleep 8
  adb exec-out screencap -p > "$out/page-$page.png"
done
python3 - "$out" <<'PY' || true
import base64, io, sys
from pathlib import Path
from PIL import Image
for path in sorted(Path(sys.argv[1]).glob('page-*.png')):
    image = Image.open(path).convert('RGB')
    image = image.resize((540, round(image.height * 540 / image.width)), Image.LANCZOS)
    buffer = io.BytesIO(); image.save(buffer, format='JPEG', quality=80, optimize=True)
    print('CC_SHOT %s %s' % (path.stem, base64.b64encode(buffer.getvalue()).decode()))
PY
echo "Android smoke test passed"
