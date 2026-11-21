package com.ccmedia.studio;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.os.Build;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.os.PowerManager;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import java.io.File;

/**
 * Runs the Python worker and GUI server. It stays in the foreground so jobs
 * and the idle release of the Kaggle session continue while the app is in the
 * background, and holds a partial wake lock only while jobs are queued or a
 * Kaggle session is held.
 */
public class StudioService extends Service {
    static final String ACTION_EXIT = "com.ccmedia.studio.EXIT";
    static final int PORT = 8787;
    private static final String CHANNEL = "studio";
    private static final int NOTIFICATION = 1;
    private static final long POLL_MS = 30_000;

    static volatile String url;
    static volatile String error;
    static volatile boolean exiting;

    private Handler handler;
    private PowerManager.WakeLock wakeLock;
    private PyObject boot;

    @Override
    public void onCreate() {
        super.onCreate();
        handler = new Handler(Looper.getMainLooper());
        NotificationManager manager = getSystemService(NotificationManager.class);
        manager.createNotificationChannel(new NotificationChannel(CHANNEL,
                getString(R.string.channel_name), NotificationManager.IMPORTANCE_LOW));
        Notification notification = notification(getString(R.string.running));
        if (Build.VERSION.SDK_INT >= 34) {
            startForeground(NOTIFICATION, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE);
        } else {
            startForeground(NOTIFICATION, notification);
        }
        PowerManager power = getSystemService(PowerManager.class);
        wakeLock = power.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "ccmedia:jobs");
        wakeLock.setReferenceCounted(false);
        new Thread(this::startPython, "cc-media-start").start();
    }

    private void startPython() {
        try {
            if (!Python.isStarted()) Python.start(new AndroidPlatform(this));
            File app = AppFiles.prepare(this);
            boot = Python.getInstance().getModule("android_boot");
            url = boot.callAttr("start", app.getPath(), new File(getFilesDir(), "data").getPath(), PORT).toString();
            handler.post(this::poll);
        } catch (Throwable failure) {
            error = failure.toString();
        }
    }

    private void poll() {
        if (exiting || boot == null) return;
        new Thread(() -> {
            boolean busy = false;
            try {
                busy = boot.callAttr("busy").toBoolean();
            } catch (Throwable ignored) {
            }
            if (busy) wakeLock.acquire(POLL_MS * 3);
            else if (wakeLock.isHeld()) wakeLock.release();
            handler.postDelayed(this::poll, POLL_MS);
        }, "cc-media-poll").start();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null && ACTION_EXIT.equals(intent.getAction()) && !exiting) {
            exiting = true;
            getSystemService(NotificationManager.class).notify(NOTIFICATION, notification(getString(R.string.releasing)));
            MainActivity.finishAll();
            new Thread(() -> {
                try {
                    if (boot != null) boot.callAttr("stop");
                } catch (Throwable ignored) {
                }
                handler.post(() -> {
                    if (wakeLock.isHeld()) wakeLock.release();
                    stopForeground(STOP_FOREGROUND_REMOVE);
                    stopSelf();
                    // Python threads cannot be restarted in-process; end the process.
                    android.os.Process.killProcess(android.os.Process.myPid());
                });
            }, "cc-media-exit").start();
        }
        // Not sticky: a restart from the background may not start a foreground service.
        return START_NOT_STICKY;
    }

    @Override
    public void onTimeout(int startId, int foregroundServiceType) {
        stopSelf();
    }

    private Notification notification(String text) {
        PendingIntent open = PendingIntent.getActivity(this, 0, new Intent(this, MainActivity.class),
                PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);
        PendingIntent exit = PendingIntent.getService(this, 1, new Intent(this, StudioService.class).setAction(ACTION_EXIT),
                PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);
        return new Notification.Builder(this, CHANNEL)
                .setSmallIcon(R.drawable.ic_notification)
                .setContentTitle(getString(R.string.app_name))
                .setContentText(text)
                .setContentIntent(open)
                .setOngoing(true)
                .addAction(new Notification.Action.Builder(null, getString(R.string.exit), exit).build())
                .build();
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}
