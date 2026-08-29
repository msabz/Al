@echo off
set GRADLE_VERSION=7.5.1
set CACHE_DIR=%USERPROFILE%\.gradle\wrapper\dists\equation-solver-%GRADLE_VERSION%
set ZIP=%CACHE_DIR%\gradle-%GRADLE_VERSION%-bin.zip
set DIST=%CACHE_DIR%\gradle-%GRADLE_VERSION%
if not exist "%DIST%\bin\gradle.bat" (
  if not exist "%CACHE_DIR%" mkdir "%CACHE_DIR%"
  if not exist "%ZIP%" powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://services.gradle.org/distributions/gradle-%GRADLE_VERSION%-bin.zip' -OutFile '%ZIP%'"
  powershell -NoProfile -Command "Expand-Archive -Force '%ZIP%' '%CACHE_DIR%'"
)
call "%DIST%\bin\gradle.bat" %*
