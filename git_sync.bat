@echo off
echo ========================================================
echo  AUTO GIT SYNC
echo ========================================================

echo 1. Adding all changes...
git add .

echo 2. Committing changes...
set /p commit_msg="Enter commit message (default: 'Auto sync from local'): "
if "%commit_msg%"=="" set commit_msg=Auto sync from local
git commit -m "%commit_msg%"

echo 3. Pushing to remote...
git push

echo ========================================================
echo  DONE!
echo ========================================================
pause
