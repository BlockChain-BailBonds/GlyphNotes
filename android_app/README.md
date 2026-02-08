# Android APK (Standalone Bail Bonds)

This directory contains a minimal Android app scaffold that mirrors the standalone bail-bonds workflow in Kotlin.

## What it does

- Provides a single **"I'm going to jail"** button.
- Creates a provisional case immediately and renders an instant response.
- Populates Tulsa court/jail metadata, deterministic risk score, and recovery gating.
- Keeps the UI flow unchanged (one action, async enrichment).

## Build

From the `android_app` directory:

```bash
./gradlew assembleDebug
```

The APK will be generated under `app/build/outputs/apk/debug/`.

> The Gradle wrapper is not included in this repository. If you do not already have it, generate it with
> `gradle wrapper` or use Android Studio to sync the project and build the APK.
