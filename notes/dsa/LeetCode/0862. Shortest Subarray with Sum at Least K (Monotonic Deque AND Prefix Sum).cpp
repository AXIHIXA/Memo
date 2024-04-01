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

        for (int i = 0; i <= n; i++)
        {
			while (!deq.empty() && k <= ps[i] - ps[deq.front()])
            {
				// 如果当前的前缀和 - 头前缀和，达标！
				ans = std::min(ans, i - deq.front());
                deq.pop_front();
			}

			// 前i个数前缀和，从尾部加入
			// 小 大
			while (!deq.empty() && ps[i] <= ps[deq.back()])
            {
				deq.pop_back();
			}

			deq.emplace_back(i);
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