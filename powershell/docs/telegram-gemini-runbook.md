# Runbook do Gateway Telegram/Gemini

Este runbook descreve a operação do gateway Telegram/Gemini usado no projeto Report Preview.

## Componentes envolvidos

- Script de inicialização: [../start_telegram_gemini_gateway.ps1](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/powershell/start_telegram_gemini_gateway.ps1)
- Script de reinício: [../restart_telegram_gemini_gateway.ps1](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/powershell/restart_telegram_gemini_gateway.ps1)
- Script de parada: [../stop_telegram_gemini_gateway.ps1](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/powershell/stop_telegram_gemini_gateway.ps1)
- Processo principal: [../telegram_gemini_gateway.py](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/telegram_gemini_gateway.py)
- Wrapper de consulta: [../ask_gemini_with_hermes_memory.ps1](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/ask_gemini_with_hermes_memory.ps1)

## O que o start faz

Ao executar o script de start, o fluxo é:

1. Validar o `ProjectRoot` e localizar `python.exe` na `.venv`.
2. Validar a existência de `telegram_gemini_gateway.py`.
3. Validar o caminho do `HermesWorkspace`.
4. Entrar no workspace Hermes e rodar `hermes gateway stop`.
5. Iniciar o gateway Python em uma nova janela PowerShell.
6. Salvar o PID em `outputs/hermes/chat/telegram_gemini_gateway.pid`.

## Comando padrão

```powershell
powershell -ExecutionPolicy Bypass -File "C:\Users\Boanerges\Desktop\Projetos\Report Preview\powershell\start_telegram_gemini_gateway.ps1" -HermesWorkspace "E:\Hermes_Workspace" -GeminiModel "gemini-2.5-flash"
```

## Quando usar cada script

### Start
Use [../start_telegram_gemini_gateway.ps1](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/powershell/start_telegram_gemini_gateway.ps1) quando o gateway Telegram/Gemini estiver desligado.

### Stop
Use [../stop_telegram_gemini_gateway.ps1](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/powershell/stop_telegram_gemini_gateway.ps1) quando precisar encerrar o processo sem mexer no gateway Hermes ou antes de manutenção local.

### Restart
Use [../restart_telegram_gemini_gateway.ps1](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/powershell/restart_telegram_gemini_gateway.ps1) após troca de configuração, alteração de token, mudança de modelo ou ajuste no script Python.

## Verificações pós-start

### Confirmar arquivo de PID

```powershell
Get-Content "C:\Users\Boanerges\Desktop\Projetos\Report Preview\outputs\hermes\chat\telegram_gemini_gateway.pid"
```

### Confirmar processo ativo

```powershell
Get-Process -Id (Get-Content "C:\Users\Boanerges\Desktop\Projetos\Report Preview\outputs\hermes\chat\telegram_gemini_gateway.pid")
```

### Acompanhar log

```powershell
Get-Content "C:\Users\Boanerges\Desktop\Projetos\Report Preview\outputs\hermes\chat\telegram_gemini_gateway.log" -Tail 50
```

## Falhas comuns

### `HermesWorkspace nao encontrado`
O caminho informado em `-HermesWorkspace` não existe na máquina atual.

Ação:
- confirmar se o HD externo está montado;
- confirmar a letra do drive;
- passar o caminho correto em `-HermesWorkspace`.

### `Python nao encontrado`
A `.venv` do projeto não foi criada ou o `ProjectRoot` está incorreto.

Ação:
- validar `C:\Users\Boanerges\Desktop\Projetos\Report Preview\.venv\Scripts\python.exe`;
- recriar a venv, se necessário.

### Timeout ao processar consulta
A consulta ao wrapper [../ask_gemini_with_hermes_memory.ps1](c:/Users/Boanerges/Desktop/Projetos/Report%20Preview/ask_gemini_with_hermes_memory.ps1) excedeu o tempo configurado.

Ação:
- revisar carga do modelo Gemini;
- revisar prompt e tamanho do contexto;
- verificar latência do modelo e do ambiente Hermes.

### `getaddrinfo failed`
Falha de resolução DNS ou conectividade ao acessar a API do Telegram.

Ação:
- validar internet e DNS da máquina;
- testar resolução do host do Telegram;
- revisar proxy, firewall ou VPN.

## Operação recomendada

Sequência segura para subir o serviço:

1. Confirmar que o workspace Hermes existe.
2. Confirmar que a `.venv` do projeto existe.
3. Rodar o script de start.
4. Ler o log do gateway.
5. Enviar uma mensagem de teste ao bot no Telegram.

## Comandos úteis

### Verificar o workspace Hermes

```powershell
Test-Path "E:\Hermes_Workspace"
```

### Verificar drives disponíveis

```powershell
Get-PSDrive
```

### Reiniciar tudo rapidamente

```powershell
powershell -ExecutionPolicy Bypass -File "C:\Users\Boanerges\Desktop\Projetos\Report Preview\powershell\restart_telegram_gemini_gateway.ps1" -HermesWorkspace "E:\Hermes_Workspace"
```
