# Windows

## 🌱 Shortcuts

- Adjust window size (full, half screen, etc.): `Win`+`Left/Right/Up/Down`. 

## 🌱 Chocolately Package Manager

```powershell
choco install vim
```

## 🌱 WSL 2 

- [Enable WSL 2](https://docs.microsoft.com/zh-cn/windows/wsl/install-win10#step-1---download-the-linux-kernel-update-package)
- [Windows Terminal](https://docs.microsoft.com/zh-cn/windows/terminal/)

## 🌱 Environment Variables

- Search > "Edit the system environment variables".
- Do **NOT** use "Edit the environment variables for your account", as the system variables will gray out!

## 🌱 Docker Desktop

- Download from the Docker website and simply manually install Docker Desktop. 
- Needs manual start from the GUI every startup.
- Desktop tray > Docker Desktop > Switch to Windows Container. 

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

### Vimrc

```bash
code ~/.vimrc

" ===== Basic Settings =====
syntax on                      " Enable syntax highlighting
filetype plugin indent on      " Enable file type detection
"set number                     " Show line numbers
"set relativenumber             " Relative line numbers
set ruler                      " Show cursor position
set showcmd                    " Show command in status bar
"set showmode                   " Show current mode
"set cursorline                 " Highlight current line

" ===== Colors =====
set t_Co=256                   " Use 256 colors
"set background=dark            " Dark background
"colorscheme desert             " Color scheme

" ===== Indentation =====
set autoindent                 " Auto-indent new lines
set smartindent                " Smart indentation
set tabstop=4                  " Tab width
set shiftwidth=4               " Indent width
set expandtab                  " Use spaces instead of tabs

" ===== Search =====
set hlsearch                   " Highlight search results
set incsearch                  " Incremental search
set ignorecase                 " Ignore case in search
set smartcase                  " Case-sensitive if uppercase present

" ===== Performance =====
set lazyredraw                 " Don't redraw during macros
set ttyfast                    " Faster terminal

" ===== Backup =====
set nobackup                   " No backup files
set noswapfile                 " No swap files
```

## 🌱 PowerShell

- [PowerShell](../powershell/powershell.md).
