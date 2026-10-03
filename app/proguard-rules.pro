# Referenced by app/build.gradle. Empty is fine while minifyEnabled is false,
# but the file must exist or the Gradle sync fails.

# TFLite reflects into these; keep them if minification is ever turned on.
-keep class org.tensorflow.lite.** { *; }
-keep class org.tensorflow.lite.nnapi.** { *; }
