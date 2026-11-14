package com.ccmedia.studio;

import android.content.Context;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/** Extracts the Python client (assets/studio) to private storage once per install or update. */
final class AppFiles {
    private AppFiles() {}

    static File prepare(Context context) throws IOException {
        long stamp;
        try {
            PackageInfo info = context.getPackageManager().getPackageInfo(context.getPackageName(), 0);
            stamp = info.lastUpdateTime;
        } catch (PackageManager.NameNotFoundException error) {
            throw new IOException(error);
        }
        File target = new File(context.getFilesDir(), "studio-" + stamp);
        File marker = new File(target, ".complete");
        if (marker.isFile()) return target;
        File[] old = context.getFilesDir().listFiles((dir, name) -> name.startsWith("studio-"));
        if (old != null) for (File folder : old) delete(folder);
        copy(context.getAssets(), "studio", target);
        if (!marker.createNewFile() && !marker.isFile()) throw new IOException("Cannot mark " + target);
        return target;
    }

    private static void copy(AssetManager assets, String path, File target) throws IOException {
        String[] children = assets.list(path);
        if (children == null || children.length == 0) {
            File parent = target.getParentFile();
            if (parent != null && !parent.isDirectory() && !parent.mkdirs()) throw new IOException("Cannot create " + parent);
            try (InputStream in = assets.open(path); OutputStream out = new FileOutputStream(target)) {
                byte[] buffer = new byte[64 * 1024];
                int read;
                while ((read = in.read(buffer)) > 0) out.write(buffer, 0, read);
            }
            return;
        }
        if (!target.isDirectory() && !target.mkdirs()) throw new IOException("Cannot create " + target);
        for (String child : children) copy(assets, path + "/" + child, new File(target, child));
    }

    private static void delete(File file) {
        File[] children = file.listFiles();
        if (children != null) for (File child : children) delete(child);
        //noinspection ResultOfMethodCallIgnored
        file.delete();
    }
}
