package com.ccmedia.studio;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.net.Uri;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.util.Base64;
import android.webkit.JavascriptInterface;
import android.widget.Toast;

import java.io.IOException;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.Map;

/**
 * Saves files that the pages offer as downloads. WebView ignores
 * {@code <a download>}, so cc_bridge.js streams the bytes here in base64
 * chunks and they are written to Downloads/CC Media Studio.
 */
final class DownloadBridge {
    private static final String FOLDER = "CC Media Studio";

    private final Context context;
    private final Handler main = new Handler(Looper.getMainLooper());
    private final Map<Integer, Pending> open = new HashMap<>();
    private int next = 1;
    private volatile boolean trusted;

    private static final class Pending {
        final Uri uri;
        final OutputStream out;
        final String name;

        Pending(Uri uri, OutputStream out, String name) {
            this.uri = uri;
            this.out = out;
            this.name = name;
        }
    }

    DownloadBridge(Context context) {
        this.context = context.getApplicationContext();
    }

    /** Only pages served by the local studio may write files. */
    void setTrusted(boolean value) {
        trusted = value;
    }

    static String safeName(String name) {
        String clean = name == null ? "" : name.replaceAll("[\\\\/:*?\"<>|\\p{Cntrl}]", "_").trim();
        if (clean.isEmpty() || clean.startsWith(".")) clean = "download" + clean;
        return clean.length() > 120 ? clean.substring(clean.length() - 120) : clean;
    }

    @JavascriptInterface
    public synchronized int begin(String name, String mime) {
        if (!trusted) return -1;
        String file = safeName(name);
        ContentValues values = new ContentValues();
        values.put(MediaStore.MediaColumns.DISPLAY_NAME, file);
        values.put(MediaStore.MediaColumns.MIME_TYPE, mime == null || mime.isEmpty() ? "application/octet-stream" : mime);
        values.put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS + "/" + FOLDER);
        values.put(MediaStore.MediaColumns.IS_PENDING, 1);
        ContentResolver resolver = context.getContentResolver();
        Uri uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values);
        if (uri == null) return -1;
        try {
            OutputStream out = resolver.openOutputStream(uri);
            if (out == null) throw new IOException("No output stream");
            int id = next++;
            open.put(id, new Pending(uri, out, file));
            return id;
        } catch (IOException failure) {
            resolver.delete(uri, null, null);
            return -1;
        }
    }

    @JavascriptInterface
    public synchronized boolean chunk(int id, String base64) {
        Pending pending = open.get(id);
        if (pending == null) return false;
        try {
            pending.out.write(Base64.decode(base64, Base64.DEFAULT));
            return true;
        } catch (IOException | IllegalArgumentException failure) {
            return false;
        }
    }

    @JavascriptInterface
    public synchronized boolean finish(int id) {
        Pending pending = open.remove(id);
        if (pending == null) return false;
        try {
            pending.out.close();
            ContentValues values = new ContentValues();
            values.put(MediaStore.MediaColumns.IS_PENDING, 0);
            context.getContentResolver().update(pending.uri, values, null, null);
            toast(context.getString(R.string.saved, FOLDER + "/" + pending.name));
            return true;
        } catch (IOException failure) {
            context.getContentResolver().delete(pending.uri, null, null);
            return false;
        }
    }

    @JavascriptInterface
    public synchronized void abort(int id) {
        Pending pending = open.remove(id);
        if (pending == null) return;
        try {
            pending.out.close();
        } catch (IOException ignored) {
        }
        context.getContentResolver().delete(pending.uri, null, null);
    }

    @JavascriptInterface
    public void failed(String reason) {
        toast(context.getString(R.string.save_failed, reason == null ? "" : reason));
    }

    private void toast(String text) {
        main.post(() -> Toast.makeText(context, text, Toast.LENGTH_LONG).show());
    }
}
