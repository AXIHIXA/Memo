# _Effective C++_ Notes

- Notes of reading <u>Scott Meyers</u>'s _Effective C++_ series:
    1. *[`Effective C++`](https://github.com/AXIHIXA/Memo/blob/master/notes/cpp/effective_cpp_notes_01_effective_cpp.md)*
    2. *[`More Effective C++`](https://github.com/AXIHIXA/Memo/blob/master/notes/cpp/effective_cpp_notes_02_more_effective_cpp.md)*
    3. ***[`Effective STL`](https://github.com/AXIHIXA/Memo/blob/master/notes/cpp/effective_cpp_notes_03_effective_stl.md)***
    4. *[`Effective Modern C++`](https://github.com/AXIHIXA/Memo/blob/master/notes/cpp/effective_cpp_notes_04_effective_modern_cpp.md)*






---

## 🌱 _Effective STL_

### 🎯 Chapter 1. Containers

### 📌 Item 1: Choose your containers with care






### 📌 Item 2: Beware the illusion of container-independent code






### 📌 Item 3: Make copying cheap and correct for objects in containers






### 📌 Item 4: Call `empty()` instead of checking `size()` against zero






### 📌 Item 5: Prefer range member functions to their single-element counterparts






### 📌 Item 6: Be alert for C++’s most vexing parse






### 📌 Item 7: When using containers of newed pointers, remember to delete the pointers before the container is destroyed






### 📌 Item 8: Never create containers of `std::auto_ptr`s

- `std::auto_ptr` itself is already deprecated since C++11, so this item is also outdated. 






### 📌 Item 9: Choose carefully among erasing options






### 📌 Item 10: Be aware of allocator conventions and restrictions






### 📌 Item 11: Understand the legitimate uses of custom allocators






### 📌 Item 12: Have realistic expectations about the thread safety of STL containers






### 🎯 Chapter 2. `std::vector` and `std::string`

### 📌 Item 13: Prefer `std::vector` and `std::string` to dynamically allocated arrays






### 📌 Item 14: Use `reserve()` to avoid unnecessary reallocations






### 📌 Item 15: Be aware of variations in `std::string` implementations






### 📌 Item 16: Know how to pass `std::vector` and `std::string` data to legacy APIs






### 📌 Item 17: Use “the `swap` trick” to trim excess capacity






### 📌 Item 18: Avoid using `std::vector<bool>`






### 🎯 Chapter 3. Associative Containers

### 📌 Item 19: Understand the difference between equality and equivalence






### 📌 Item 20: Specify comparison types for associative containers of pointers






### 📌 Item 21: Always have comparison functions return `false` for equal values






### 📌 Item 22: Avoid in-place key modification in `std::set` and `std::multiset`






### 📌 Item 23: Consider replacing associative containers with sorted `std::vector`s






### 📌 Item 24: Choose carefully between `std::map::operator[]` and `std::map::insert` when efficiency is important






### 📌 Item 25: Familiarize yourself with the nonstandard hashed containers






### 🎯 Chapter 4. Iterators

### 📌 Item 26: Prefer `iterator` to `const_iterator`, `reverse_iterator`, and `const_reverse_iterator`

- Deprecated. Refer to _Effective Modern C++_ Item 13 for details. 






### 📌 Item 27: Use `distance()` and `advance()` to convert a container’s `const_iterator`s to `iterator`s






### 📌 Item 28: Understand how to use a `reverse_iterator`’s base `iterator`






### 📌 Item 29: Consider `std::istreambuf_iterator`s for character-by-character input






### 🎯 Chapter 5. Algorithms

### 📌 Item 30: Make sure destination ranges are big enough






### 📌 Item 31: Know your sorting options






### 📌 Item 32: Follow `remove`-like algorithms by erase if you really want to remove something






### 📌 Item 33: Be wary of `remove`-like algorithms on containers of pointers






### 📌 Item 34: Note which algorithms expect sorted ranges






### 📌 Item 35: Implement simple case-insensitive string comparisons via `mismatch` or `lexicographical_compare`






### 📌 Item 36: Understand the proper implementation of `std::copy_if`






### 📌 Item 37: Use `std::accumulate` or `std::for_each` to summarize ranges






### 🎯 Chapter 6. Functors, Functor Classes, Functions, etc. 

### 📌 Item 38: Design functor classes for pass-by-value






### 📌 Item 39: Make predicates pure functions






### 📌 Item 40: Make functor classes adaptable






### 📌 Item 41: Understand the reasons for `std::ptr_fun`, `std::mem_fun`, and `std::mem_fun_ref`






### 📌 Item 42: Make sure `std::less<T>` means `operator<`






### 🎯 Chapter 7. Programming with the STL 

### 📌 Item 43: Prefer algorithm calls to hand-written loops






### 📌 Item 44: Prefer member functions to algorithms with the same names






### 📌 Item 45: Distinguish among `std::count`, `std::find`, `std::binary_search`, `std::lower_bound`, `std::upper_bound`, and `std::equal_range`






### 📌 Item 46: Consider function objects instead of functions as algorithm parameters






### 📌 Item 47: Avoid producing write-only code






### 📌 Item 48: Always `#include` the proper headers






### 📌 Item 49: Learn to decipher STL-related compiler diagnostics






### 📌 Item 50: Familiarize yourself with STL-related web sites




