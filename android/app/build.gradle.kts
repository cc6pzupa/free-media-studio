plugins {
    id("com.android.application")
    id("com.chaquo.python")
}

// The Python client and its pages are copied from the repository root into the
// APK assets; the app extracts them to private storage on first launch.
val repositoryRoot = rootProject.projectDir.parentFile
val studioAssets = layout.buildDirectory.dir("generated/studioAssets")
val copyStudio by tasks.registering(Sync::class) {
    from(repositoryRoot) {
        include("*.py", "*.html")
        exclude("build_*.py", "*benchmark*.py", "*_gpu_check.py", "test_*.py")
    }
    into(studioAssets.map { it.dir("studio") })
}

// Release signing comes from CI secrets; without them the release APK uses the debug key.
val keystorePath: String? = System.getenv("ANDROID_KEYSTORE_PATH")

android {
    namespace = "com.ccmedia.studio"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.ccmedia.studio"
        minSdk = 29
        targetSdk = 35
        versionCode = 1000
        versionName = "0.10.0"
        ndk {
            // Phones are arm64; x86_64 is for the emulator smoke test.
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    signingConfigs {
        if (keystorePath != null) {
            create("release") {
                storeFile = file(keystorePath)
                storePassword = System.getenv("ANDROID_KEYSTORE_PASSWORD")
                keyAlias = System.getenv("ANDROID_KEY_ALIAS")
                keyPassword = System.getenv("ANDROID_KEY_PASSWORD")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.findByName("release") ?: signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    sourceSets["main"].assets.srcDir(studioAssets)
}

tasks.named("preBuild") { dependsOn(copyStudio) }

chaquopy {
    defaultConfig {
        version = "3.12"
        pip {
            // Local client dependencies only (requirements.txt minus remote/desktop packages).
            install("requests==2.34.2")
            install("websocket-client==1.9.2")
            install("six==1.17.0")
        }
    }
}
