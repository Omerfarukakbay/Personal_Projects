@echo off
REM Usage: scripts\run.bat [create|sim|build|program|all]
setlocal
cd /d "%~dp0.."
set T=%1
if "%T%"=="" set T=all
if /i "%T%"=="create"  goto create
if /i "%T%"=="sim"     goto sim
if /i "%T%"=="build"   goto build
if /i "%T%"=="program" goto program
if /i "%T%"=="all"     goto all
echo Unknown target %T% & exit /b 1
:all
call :create || exit /b 1
call :build  || exit /b 1
goto program
:create
if not exist build mkdir build
vivado -mode batch -nojournal -nolog -source scripts\create_project.tcl -tempDir build & exit /b %errorlevel%
:sim
vivado -mode batch -nojournal -nolog -source scripts\sim.tcl -tempDir build & exit /b %errorlevel%
:build
vivado -mode batch -nojournal -nolog -source scripts\build.tcl -tempDir build & exit /b %errorlevel%
:program
vivado -mode batch -nojournal -nolog -source scripts\program.tcl -tempDir build & exit /b %errorlevel%
