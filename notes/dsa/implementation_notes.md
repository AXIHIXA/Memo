# Implementation Notes



## String Concatenation

- While serializing one should avoid concatenating strings in programming languages where strings are immutable 
  - Concatenating strings takes linear time. 
  - If we concatenate a string of length `O(N)` about `O(N)` times, then the total time complexity will be `O(N**2)`.
- Instead, one should use a `StringBuilder` or `StringBuffer` in `Java`, or `list` in `Python`. 
- Now, `std::string` in `C++` is mutable, so we can concatenate strings. 
  - `operator +` **wouldn't** create a new string each time. 
  - It would just append the new string to the existing string. 
  - Therefore, we can use `operator +` to concatenate strings in `C++`.
- [Reference](https://leetcode.com/problems/subtree-of-another-tree/editorial/)



