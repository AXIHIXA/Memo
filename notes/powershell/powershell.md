# PowerShell

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



