# PowerShell

## 🌱 PowerShell Configuration

### PSReadlineOptions

- Set PS readline options into Emacs style:
```powershell
code $profile

# Use Emacs (Bash) keybindings
Set-PSReadLineOption -EditMode Emacs
```
- By default:
  - Move cursor to front/back: Windows `Home/End` vs. Emacs `Ctrl+A/E`.
  - Clear content before/after cursor: Windows `Ctrl+Home/End` vs. Emacs `Ctrl+U/K`.
  - Move cursor forward/backward by one word: Windows `Ctrl+LeftArrow/RightArrow` vs. Emacs `Alt+LeftArrow/RightArrow`.
  - Delete the word before/after the cursor: Windows `Ctrl+Backspace/Delete` vs. Emacs `Ctrl+W`/`Alt+D`.

### Migration from Bash

#### 1. Save Configurations

- PowerShell `write-host` vs. Bash `echo`.
  - In PowerShell, `write-host` is a wrapper of `write-output`, `echo` is an alias to `write-output`. 
- PowerShell `$profile` vs. Bash `~/.profile/` & `~/.bashrc`.
```powershell
PS C:\Users\xihan> write-host $profile
C:\Users\xihan\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1
PS C:\Users\xihan> $profile
C:\Users\xihan\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1
```

#### 2. Core Variable & Symbol Mapping

| Feature | Bash | PowerShell |
| :--- | :--- | :--- |
| **Variable Syntax** | `var="val"` | `$var = "val"` |
| **Home Directory** | `~` | `$HOME` or `~` or `$env:USERPROFILE` |
| **Current Directory** | `$PWD` | `$PWD` |
| **Last Command Success** | `$?` | `$?` (Returns True/False) |
| **External Exit Code** | `$?` | `$LASTEXITCODE` (Returns Number) |
| **Env Variables** | `$PATH` | `$env:PATH` |
| **Process ID** | `$$` | `$PID` |

#### 3. Command Equivalents

| Bash Command | PowerShell Cmdlet | Common Alias |
| :--- | :--- | :--- |
| `echo "text"` | `Write-Output "text"` | `echo`, `write` |
| `ls -al` | `Get-ChildItem` | `ls`, `dir`, `gci` |
| `grep "pattern"` | `Select-String "pattern"` | `sls` |
| `cat file.txt` | `Get-Content file.txt` | `cat`, `type`, `gc` |
| `mkdir folder` | `New-Item -Type Directory` | `mkdir` |
| `rm -rf` | `Remove-Item -Recurse -Force` `rm -Resurse -Force` `rm -r -fo` | `rm`, `del`, `ri` |
| `find . -name "*.js"` | `Get-ChildItem -Recurse -Filter *.js` | `gci -r` |
| `man command` | `Get-Help command -Online` | `help` |

#### 4. Logic & Operators

| Feature | Bash | PowerShell |
| :--- | :--- | :--- |
| **String Comparison** | `==`, `!=` | `-eq`, `-ne` |
| **Numeric Comparison** | `-gt`, `-lt`, `-ge`, `-le` | `-gt`, `-lt`, `-ge`, `-le` |
| **AND / OR** | `&&` / `||` | `&&` / `||` (PS 7+) or `-and` / `-or` |
| **Pipe** | `\|` (Passes Text) | `\|` (Passes **Objects**) |
| **Redirection** | `>` or `>>` | `>` or `>>` (Defaults to UTF-16) |



## 🌱 PowerShell Overview

- Windows PowerShell Syntax:
  - `Verb-Noun -NameParameter ArgumentString`.
  - E.g., `Get-Service -Name "*net*"`.
  - In most cases, case **insensitive**.
  - In most cases, most commands are singular instead of plural. 
- Get Help (Windows Version of `man`):
  - `Get-Help COMMAND`.
  - Do **not** use `Update-Help` for the "help file". The "detailed version" is worse than the "partial help". 
- Pipe: `|`
```powershell
> Get-Help New-Item
> New-Item -Path "$Home/.ssh" -ItemType "Directory" 
> New-Item -Path "$Home/.ssh/authorized_keys" -Value "xxx"
> Get-Content -Path "$Home/.ssh/authorized_keys"
> Set-Content -Path "xxx" -Value "xxx"
```
- Cmdlets and Aliases
  - `dir`/`ls`/`gci`: Aliases for `Get-ChildItem`
- Pipe and Filter
```powershell
> Get-Service | Where-Object {$_.status -eq "Stopped"}
> Get-Service Get-Member
```

## 🌱 Using PowerShell

### Functions

```powershell
> function add
>>> {
>>> $add = [int](2 + 2)
>>> Write-Output "$add"  
>>> }
> add
4
```

## Parameter `-WhatIf` and `Confirm`

```powershell
# Dry run:
> Get-Service | Stop-Service -WhatIf
# Confirm Y/N one by one for piped input:
> Get-Service | Stop-Service -Confirm
```

## Working with Output

```powershell
# Pipe formatting:
> Get-Service | Format-List DisplayName, Status, RequiredServices
> Get-Service | Format-List *  # Wildcard, display everything. 
> Get-Service | Sort-Object -Property Status | Format-List 

# Pipe output:
> Get-Service | Out-File C:\services.txt
> Get-Service | Export-Csv C:\services.csv
```

## Running PowerShell Remotely

```powershell
> Get-Service -ComputerName MyWebServer, AnotherWebServer | Sort-Object ...
```



