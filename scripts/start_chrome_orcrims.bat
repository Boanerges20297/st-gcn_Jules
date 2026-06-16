@echo off
REM Abre o Chrome no perfil padrao (Default = boanergesteixeiraalmeida@gmail.com)
REM com a porta de depuracao ativa, para que a automacao de ORCRIMs consiga
REM capturar os cookies da sessao viva e baixar o KMZ do MyMaps via HTTP.
REM
REM Use ESTE atalho no lugar do Chrome normal quando quiser que a automacao
REM funcione com o navegador aberto. Visualmente o Chrome e identico.
REM
REM Importante: feche todas as janelas do Chrome desse perfil antes da
REM primeira vez, senao a flag de depuracao e ignorada (o Chrome ja rodando
REM apenas repassa o comando para a instancia existente).

set "CHROME=C:\Program Files\Google\Chrome\Application\chrome.exe"
set "USERDATA=%LOCALAPPDATA%\Google\Chrome\User Data"

start "" "%CHROME%" ^
  --user-data-dir="%USERDATA%" ^
  --profile-directory=Default ^
  --remote-debugging-port=9222 ^
  --remote-allow-origins=*
