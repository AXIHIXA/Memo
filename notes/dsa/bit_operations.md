# Bit Operations

## Bitwise AND

- `x & (x - 1)`: Unset lowest 1 bit;
  - Usage: Bin count, etc. 
- `x & (-x)` or `x & (~x + 1)`: Extract lowest 1 bit (unset all other bits);
- If `n` is power of 2, then `x % n` equals `x & (n - 1)`. 

## Bitwise XOR

- Carry-less addition. 
  - Communicative; 
  - Associative. 
- XOR zero equals self: `x ^ 0 == x`;
- XOR self equals zero: `x ^ x == 0`;
- `swap`: `a ^= b, b ^= a, a ^= b;` 
  - Note that this expression is wrong when `&a == &b`!
- Missing Number: `a[0:n]`, ranging from 0 to n, missing one number. 
  - `n ^ (i ^ a[i] for i in range(n))`
- One number odd occurance, all other numbers even occurance, find this number: 
  - XOR all elements in array and the result is this odd-occurance number;
- Two numbers odd occurence, all other numbers even occurence, find these two numbers:
  - XOR all elements in array, extract lowest 1 bit in the result;
  - The elements in this array could be divided in two groups (checking this extracted bit);
  - Each group contains one of the two odd-occurance numbers. 
- One number occurs less than m times, all other numbers occur exactly m times, find this number:
  - Calculate the number of 1-bits for bit 0 to 31 for all elements in this array;
  - Extract bits whose number of 1-bits is not multiple of m;
  - The result is the extracted number. 

## Unrolled Bin Count

```c++
int binCount(unsigned int n)
{
    n = (n & 0x55555555U) + ((n & 0xaaaaaaaaU) >> 1U);
    n = (n & 0x33333333U) + ((n & 0xccccccccU) >> 2U);
    n = (n & 0x0f0f0f0fU) + ((n & 0xf0f0f0f0U) >> 4U);
    n = (n & 0xff00ffU)   + ((n & 0xff00ff00U) >> 8U);
    n = (n & 0xffffU)     + ((n & 0xffff0000U) >> 16U);
    return n;
}
```
