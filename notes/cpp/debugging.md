# Debugging with gdb/lldb

## Multithreading Deadlock Debugging

```bash
$ sudo ps
# Get PID of deadlocked process
$ gdb
# Now in gdb session 
(gdb) attach <PID>

(gdb) help
(gdb) help thread

(gdb) thread backtrace all
# Output: Threads and frames

(gdb) t 1
# Selected thread #1
(gdb) frame variables
(gdb) p uniquePtr.get()->member
```

- `p` command in gdb could not handle smart pointers (?), get their base built-in pointers for further processing. 
