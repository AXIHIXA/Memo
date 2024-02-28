class Solution
{
public:
    int kthGrammar(int n, int k)
    {
        return dfs(n, k, 1) == 0 ? 1 : 0;
    }

private:
    // Given that k-th digit in n-th row is cur, what is the digit on row 1?
    static int dfs(int n, int k, int cur)
    {
        if (n == 1) return cur;
        
        if (k & 1)
        {
            // 123    123456
            // 000 -> 010101
            // 111 -> 101010
            return dfs(n - 1, 1 + (k >> 1), cur);
        }
        else
        {
            return dfs(n - 1, k >> 1, !cur);
        }
    }
};