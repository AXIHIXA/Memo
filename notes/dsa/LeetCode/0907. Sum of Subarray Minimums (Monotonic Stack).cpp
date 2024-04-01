class Solution
{
public:
    int sumSubarrayMins(std::vector<int> & arr)
    {
        auto n = static_cast<const int>(arr.size());
        std::vector<int> stk;
        stk.reserve(n);
        long long ans = 0LL;

        for (int i = 0; i < n; ++i)
        {
            while (!stk.empty() && arr[i] <= arr[stk.back()])
            {
                int top = stk.back();
                stk.pop_back();
                int ll = stk.empty() ? -1 : stk.back();
                int rr = i;

                // arr[ll]: left nearest element < arr[top];
                // arr[rr]: right neatest element < arr[top];
                // arr[top] appears as min(b) in  subarrays. 
                ans = (ans + static_cast<long long>(top - ll) * (rr - top) * arr[top]) % p;
            }

            stk.emplace_back(i);
        }

        while (!stk.empty())
        {
            int top = stk.back();
            stk.pop_back();
            int ll = stk.empty() ? -1 : stk.back();
            int rr = n;

            // arr[ll]: left nearest element < arr[top];
            // arr[rr]: right neatest element < arr[top];
            // arr[top] appears as min(b) in  subarrays. 
            ans = (ans + static_cast<long long>(top - ll) * (rr - top) * arr[top]) % p;
        }

        return ans;
    }
    
private:    
    static constexpr long long p = 1'000'000'007LL;
};