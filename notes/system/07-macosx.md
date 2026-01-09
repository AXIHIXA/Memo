# Mac OSX

## Shortcuts

- [Common Shortcuts](https://support.apple.com/en-us/102650)
  - Hard-coded, non-editable, could not be overwritten by other configurations. 
  - In case of conflicts, these common shortcuts will prevail. 
- Editable Shortcuts:
  - Finder (Top left corner) > Services > Service Settings > ...
  - OR: 
  - System Settings (from Taskbar) > Keyboard > **Keyboard Shortcuts**... >  ...
  - E.g. Use F1, F2, etc keys as standard function keys

## Touchpad

- Use three fingers
  - Swipe left/right: Change screen;
  - **Mission Control**:
    - Shortcut: Ctrl + Up
    - Swipe up: Show screen management tab on the top.

## Terminal

- Profile
  - Duplicate "Clear Dark", rename to "AX", set as Default.
  - Text -> Background -> Color & Effects -> Ocacity 90%, Blur 50%
  - Text -> Font -> font size to 16
- Notes
  - Command replaces Ctrl;
  - Ctrl + C/Z remains unchanged.
  

## Finder

- Command + Delete: Delete a file;
- Command + Shift + Period(.): Show/hide hidden files;
- Select a file and press Enter: Rename a file.
- Command + C: Select a file for copy or cut, then: 
  - Command + V: Copy
  - Command + Option + V: Cut
- Create New Text File: [HERE](https://apple.stackexchange.com/questions/84309/how-to-create-a-text-file-in-a-folder)
  - Apps -> Search "Automator" -> Choose "Quick Action" -> Left side bar choose "Utilities -> Run Apple Script"
  - Edit tabs: Workflow receives "no input" in "Finder.app"
  - Image "+ Add"
  - Double click "Run AppleScript", Replace purple script with:
```
tell application "Finder"
    set txt to make new file at (the target of the front window) as alias with properties {name:"Untitled Document.txt"}
    select txt
end tell
```
  - Top-left bar -> File -> Save, Save as "New Document" (will be stored in `~/Library/Services/`).
    - Note that this name in "Services" will be final, won't change even if you update the file name in `~/Library/Services/`. 
  - Under System Settings (in Dock) -> Keyboard Shortcuts... -> Services -> General. You will see "New Document" listed with "none" as the shortcut. Double click "none", replace with `Ctrl + Option + N` or other shortcuts. 
- To always show file extensions on macOS, go to Finder > Settings (or Preferences) > Advanced and check "Show all filename extensions"

## Chrome

- Duplicate tab: Option + Shift + D.
- Refresh page: Command + R. 
