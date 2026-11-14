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
curl -fsS http://127.0.0.1:8787/health | tee "$out/health.json" | grep -q '"qwen-studio"'
page=$(curl -fsS http://127.0.0.1:8787/)
grep -q 'CC Media Studio' <<<"$page"
nonce=$(grep -o "const nonce='[^']*'" <<<"$page" | cut -d"'" -f2)
# Goes through StudioClient to the worker thread inside the app.
curl -fsS -H "X-Local-Session: $nonce" http://127.0.0.1:8787/configuration | tee "$out/configuration.json" | grep -q '"configured"'
curl -fsS http://127.0.0.1:8787/video | grep -q 'CC Media Studio'
curl -fsS http://127.0.0.1:8787/gallery | grep -q 'CC Media Studio'
# The page itself must have loaded in the WebView (not the "starting" text).
sleep 8
adb shell dumpsys activity activities | grep -q "com.ccmedia.studio/.MainActivity"
collect
if grep -q "FATAL EXCEPTION" "$out/logcat.txt"; then grep -A20 "FATAL EXCEPTION" "$out/logcat.txt"; exit 1; fi
echo "Android smoke test passed"
