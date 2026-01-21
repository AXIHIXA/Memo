# Generic Software Shortcuts

## Vim

- Go to beginning of line: `HOME`
- Go to end of line: `END`
- Scroll down to the previous page: `PAGE UP`
- Scroll down to the next page: `PAGE DOWN`

## Terminal

- Go to beginning of line: `Ctrl` + `A`
- Go to end of line: `Ctrl` + `E`
- Delete the previous word: `Ctrl` + `W`
- Delete left of the cursor on this line: `Ctrl` + `U`
- Delete right of the cursor on this line: `Ctrl` + `K`

## [Cursor](https://cursor.com/dashboard) and [Visual Studio Code](https://code.visualstudio.com/)

### Clear Recent from Welcome Page

- From Top Toolbar: File > Open Recent > Clear Recently Opened...

### Chat Window

- Toggle On/Off Chat Window: `Option` + `Command` + `B`
- Remove Chat History: "Clock" icon on the second row of the top-right corner 

### App Shortcuts

- Show Command Palette: `Command` + `Shift` + `P` or `F1`
- Toggle On/Off Terminal: `Command` + `J`
- Toggle On/Off Left Side Bar: `Command` + `B`
- Toggle On/Off Chat Window: `Option` + `Command` + `B`
- Terminal tab -> Top right corner "+v" -> "Configure Terminal Settings"

### Code Navigation Shortcuts

- Jump between Opening and Closing Braces: `Command` + `Shift` + `\`
- Select everything between a pair of a matching brace: `Ctrl` + `Shirt` + `Right Arrow`

### Remote Development Envionment

- Install the Remote - SSH (`anysphere.remote-ssh`) and Dev Containers extensions (`anysphere.remote-containers`) extensions.
- Reload Cursor by doing Ctrl+Shift+P and typing Developer: Reload Window. 
- Open the Remote Explorer extension. To the right of “SSH TARGETS”, select the gear icon and add a new entry to the SSH config file.
- To the right of “SSH TARGETS”, click the refresh button to see your entry on the list. Click on the small button in the new entry and connect to it.
- Open the dev container. The simplest way is `Command+Shift+P` and type `Dev Container: Attach to Running Container`.
- In the CMake extension, click “build”. (If the CMake extension isn't visible, try doing > Developer: Reload Window or > CMake: Configure from Ctrl+Shift+P.)
- By default, VSCode/Cursor binds F7 as the keybinding to build the currently-selected CMake target.
  - Add a keybinding such as Ctrl+Shift+F7 for “CMake: Compile Active File”. This lets you build the open .cpp file instantly.
  - Build less than the full project by clicking Set Build Target. (Or Ctrl+Shift+P and enter > CMake: Set Build Target.)

### Settings

- **Choose the "User" tab to avoid conflicts between profiles! Other profiles default to User!**
- Cursor -> Settings -> Keyboard Shortcuts -> Search "Quick Voice Chat" -> Right Click, Remove Keybinding.
  - It conflicts inside integrated Terminal to detach a Docker container (`Ctrl+P+Q`).
- Cursor -> Settings -> VS Code Settings
  - Search "font"
    - Left Scroll Menu
      - Text Editor -> Font (`@editor font`)
        - Editor: Font Size: `20`
        - Editor: Font Family: `'JetBrains Mono', Consolas, 'Courier New', monospace`
      - Features -> Debug (`@feature:debug`)
        - Debug > Console: Font Size: `20`
      - Features -> Terminal (`@feature:terminal`)
        - Terminal > Integrated: Font Family: (Defaults to Editor: Font Family's value.)
        - Terminal > Integrated: Font Size: `20` (`@feature:terminal font size`)
  - Search "minimap"
    - Editor > Minimap Enabled: Toggle On
  - Search "cursor"
    - Left Scroll Menu
      - Features -> Terminal (`@feature:terminal`)
        - Terminal > Integrated: Cursor Blinking: Toggle On (`@feature:terminal cursor blinking`)
        - Terminal > Integrated: Cursor Style: `line` (`@feature:terminal cursor style`)
