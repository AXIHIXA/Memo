class Solution
{
public:
    Solution()
    {
        static const int _ = init();
    }

    int shortestSubarray(std::vector<int> & nums, int k)
    {
        auto n = static_cast<const int>(nums.size());

        std::vector<long long> ps(n + 1, 0LL);
        std::inclusive_scan(nums.cbegin(), nums.cend(), ps.begin() + 1, std::plus<>(), 0LL);

        int ans = std::numeric_limits<int>::max();
        std::deque<int> deq;

        // nums[ll...rr],
        // k <= ps[rr + 1] - ps[ll],
        // Nearest ps[ll] <= ps[rr + 1] - k.
        // ps[l] >= ps[r], then r could pop l (l is sub-optimal), thus ascending deque.
        
        for (int rr = 0; rr <= n; ++rr)
        {
            while (!deq.empty() && ps[rr] <= ps[deq.back()])
            {
                deq.pop_back();
            }

            deq.emplace_back(rr);

            while (!deq.empty() && ps[deq.front()] <= ps[rr] - k)
            {
                ans = std::min(ans, rr - deq.front());
                deq.pop_front();
            }
        }

        return ans == std::numeric_limits<int>::max() ? -1 : ans;
    }

private:
    static int init()
    {
        std::ios_base::sync_with_stdio(false);
        std::cin.tie(nullptr);
        std::cout.tie(nullptr);
        std::setvbuf(stdin, nullptr, _IOFBF, 1 << 20);
        std::setvbuf(stdout, nullptr, _IOFBF, 1 << 20);
        return 0;
    }
};