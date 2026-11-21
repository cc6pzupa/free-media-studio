"""Chaquopy entry: put the extracted app files on sys.path and delegate."""
import sys


def start(app_dir,data_dir,port):
    if app_dir not in sys.path:sys.path.insert(0,app_dir)
    import android_entry
    return android_entry.start(data_dir,port)


def busy():
    import android_entry
    return android_entry.busy()


def stop():
    import android_entry
    return android_entry.stop()
