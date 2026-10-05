@echo off
REM Quick command-line simulation of the echo design with sim\scratch\tb_echo.v
REM Usage: scripts\sim_echo.bat         compile + run, prints PASS/FAIL
REM        scripts\sim_echo.bat gui     same, then opens the waveform in Vivado
setlocal
cd /d "%~dp0.."
if not exist build\xsim mkdir build\xsim
cd build\xsim
REM 1) xvlog: compile (parse) every Verilog file into the "work" library
call xvlog ..\..\rtl\baud_gen.v ..\..\rtl\fifo.v ..\..\rtl\uart_rx.v ..\..\rtl\uart_tx.v ..\..\rtl\uart.v ..\..\rtl\top.v ..\..\sim\scratch\tb_echo.v || exit /b 1
REM 2) xelab: elaborate (connect the hierarchy under tb_echo) into a snapshot
call xelab tb_echo -timescale 1ns/1ps -debug typical -s tb_echo_sim || exit /b 1
REM 3) xsim: run the snapshot, recording all signals to tb_echo.wdb
call xsim tb_echo_sim -tclbatch ../../sim/scratch/wave.tcl -wdb tb_echo.wdb || exit /b 1
if /i "%1"=="gui" call xsim tb_echo.wdb -gui
