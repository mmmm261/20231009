@echo off

set _task=test_sleep.exe
set _svr=E:\cppcode\test_sleep\Release\test_sleep.exe
set _des=start_algorithm.bat

:checkstart
tasklist /nh|find /i "test_sleep.exe"
echo %ERRORLEVEL%
IF ERRORLEVEL 1 goto startsvr
IF ERRORLEVEL 0 goto checkag

:startsvr
echo wuhu %%n==%_task%
echo %time% program not exists.
echo ********start program********
echo program restart at %time% ,please check log file >> restart_service.txt
echo start %_svr% > %_des%
echo exit >> %_des%
start %_des%
set/p=.<nul
for /L %%i in (1 1 10) do set /p a=.<nul&ping.exe /n 2 127.0.0.1>nul
echo .
echo Wscript.Sleep WScript].Arguments(0) >%tmp%\delay.vbs
cscript //b //nologo %tmp%\delay.vbs 10000
del %_des% /Q
echo ********start program success********
goto checkstart

:checkag
echo %time% program true, detecting after 10s.
echo Wscript.Sleep WScript.Arguments(0) >%tmp%\delay.vbs
cscript //b //nologo %tmp%\delay.vbs 10000
goto checkstart