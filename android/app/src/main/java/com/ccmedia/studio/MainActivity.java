package com.ccmedia.studio;

import android.Manifest;
import android.app.Activity;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.Insets;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowInsets;
import android.webkit.ValueCallback;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.FrameLayout;
import android.widget.TextView;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.ref.WeakReference;
import java.nio.charset.StandardCharsets;

/** Shows the local GUI served by {@link StudioService} in a WebView. */
public class MainActivity extends Activity {
    private static final int FILE_CHOOSER = 1;
    private static WeakReference<MainActivity> current = new WeakReference<>(null);

    private final Handler handler = new Handler(Looper.getMainLooper());
    private FrameLayout root;
    private WebView web;
    private TextView message;
    private ValueCallback<Uri[]> pendingFiles;
    private View fullscreen;
    private String bridgeScript;
    private Bundle savedWebState;

    static void finishAll() {
        MainActivity activity = current.get();
        if (activity != null) activity.runOnUiThread(activity::finishAndRemoveTask);
    }

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        current = new WeakReference<>(this);
        savedWebState = state == null ? null : state.getBundle("web");
        root = new FrameLayout(this);
        root.setBackgroundColor(Color.rgb(0xf5, 0xf6, 0xfa));
        // targetSdk 35 draws edge to edge; keep the page clear of the system bars and keyboard.
        root.setOnApplyWindowInsetsListener((view, insets) -> {
            if (Build.VERSION.SDK_INT >= 30) {
                Insets bars = insets.getInsets(WindowInsets.Type.systemBars() | WindowInsets.Type.ime());
                view.setPadding(bars.left, bars.top, bars.right, bars.bottom);
            } else {
                view.setPadding(insets.getSystemWindowInsetLeft(), insets.getSystemWindowInsetTop(),
                        insets.getSystemWindowInsetRight(), insets.getSystemWindowInsetBottom());
            }
            return insets;
        });
        message = new TextView(this);
        message.setText(R.string.starting);
        message.setGravity(Gravity.CENTER);
        message.setPadding(48, 48, 48, 48);
        message.setTextColor(Color.rgb(0x60, 0x64, 0x7a));
        root.addView(message, new FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
        setContentView(root);

        if (Build.VERSION.SDK_INT >= 33 && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS}, 0);
        }
        StudioService.exiting = false;
        startForegroundService(new Intent(this, StudioService.class));
        waitForServer();
    }

    private void waitForServer() {
        if (StudioService.url != null) {
            showStudio(StudioService.url);
        } else if (StudioService.error != null) {
            message.setText(getString(R.string.start_failed, StudioService.error));
        } else {
            handler.postDelayed(this::waitForServer, 150);
        }
    }

    private void showStudio(String address) {
        if (web != null) return;
        if ((getApplicationInfo().flags & ApplicationInfo.FLAG_DEBUGGABLE) != 0) WebView.setWebContentsDebuggingEnabled(true);
        bridgeScript = asset("cc_bridge.js");
        web = new WebView(this);
        WebSettings settings = web.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setSupportMultipleWindows(false);
        settings.setAllowFileAccess(false);
        settings.setAllowContentAccess(false);
        settings.setMediaPlaybackRequiresUserGesture(true);
        settings.setUserAgentString(settings.getUserAgentString() + " CCMediaStudioAndroid/0.10");
        DownloadBridge bridge = new DownloadBridge(this);
        web.addJavascriptInterface(bridge, "CCAndroid");
        web.setWebViewClient(new WebViewClient() {
            @Override
            public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
                Uri uri = request.getUrl();
                if (isLocal(uri)) return false;
                // Kaggle, GitHub and other links open in the browser; the bridge is only for local pages.
                try {
                    startActivity(new Intent(Intent.ACTION_VIEW, uri));
                } catch (ActivityNotFoundException ignored) {
                }
                return true;
            }

            @Override
            public void onPageStarted(WebView view, String page, android.graphics.Bitmap favicon) {
                bridge.setTrusted(isLocal(Uri.parse(page)));
            }

            @Override
            public void onPageFinished(WebView view, String page) {
                if (isLocal(Uri.parse(page)) && bridgeScript != null) view.evaluateJavascript(bridgeScript, null);
            }
        });
        web.setWebChromeClient(new WebChromeClient() {
            @Override
            public boolean onShowFileChooser(WebView view, ValueCallback<Uri[]> callback, FileChooserParams params) {
                if (pendingFiles != null) pendingFiles.onReceiveValue(null);
                pendingFiles = callback;
                try {
                    startActivityForResult(params.createIntent(), FILE_CHOOSER);
                } catch (ActivityNotFoundException failure) {
                    pendingFiles = null;
                    return false;
                }
                return true;
            }

            @Override
            public void onShowCustomView(View view, CustomViewCallback callback) {
                fullscreen = view;
                root.addView(view, new FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
            }

            @Override
            public void onHideCustomView() {
                if (fullscreen != null) root.removeView(fullscreen);
                fullscreen = null;
            }
        });
        root.removeView(message);
        root.addView(web, 0, new FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
        if (savedWebState == null || web.restoreState(savedWebState) == null) web.loadUrl(address);
    }

    private static boolean isLocal(Uri uri) {
        return uri != null && "http".equals(uri.getScheme()) && "127.0.0.1".equals(uri.getHost());
    }

    private String asset(String name) {
        try (InputStream in = getAssets().open(name)) {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            byte[] buffer = new byte[8192];
            int read;
            while ((read = in.read(buffer)) > 0) out.write(buffer, 0, read);
            return out.toString(StandardCharsets.UTF_8.name());
        } catch (IOException failure) {
            return null;
        }
    }

    @Override
    protected void onActivityResult(int request, int result, Intent data) {
        super.onActivityResult(request, result, data);
        if (request == FILE_CHOOSER && pendingFiles != null) {
            pendingFiles.onReceiveValue(WebChromeClient.FileChooserParams.parseResult(result, data));
            pendingFiles = null;
        }
    }

    @Override
    @SuppressWarnings("deprecation")
    public void onBackPressed() {
        if (fullscreen != null) {
            web.getWebChromeClient().onHideCustomView();
        } else if (web != null && web.canGoBack()) {
            web.goBack();
        } else {
            // Keep the service and any running job alive; exit from the notification.
            moveTaskToBack(true);
        }
    }

    @Override
    protected void onSaveInstanceState(Bundle state) {
        super.onSaveInstanceState(state);
        if (web != null) {
            Bundle webState = new Bundle();
            web.saveState(webState);
            state.putBundle("web", webState);
        }
    }

    @Override
    protected void onDestroy() {
        handler.removeCallbacksAndMessages(null);
        if (web != null) {
            web.removeJavascriptInterface("CCAndroid");
            web.destroy();
        }
        if (current.get() == this) current = new WeakReference<>(null);
        super.onDestroy();
    }
}
