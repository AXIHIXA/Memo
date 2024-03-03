class Solution
{
public:
    int nextGreaterElement(int n)
    {
        std::string s = std::to_string(n);
        nextPermutation(s.begin(), s.end());
        long long ans = std::stoll(s);
        return n < ans && ans <= std::numeric_limits<int>::max() ? ans : -1;
    }

private:
    template <typename Iter>
    static void nextPermutation(Iter b, Iter e)
    {
        auto rb = std::make_reverse_iterator(e);
        auto re = std::make_reverse_iterator(b);
        auto ll = std::is_sorted_until(rb, re);

        if (ll != re)
        {
            auto rr = std::upper_bound(rb, ll, *ll);
            std::iter_swap(ll, rr);
        }

        std::reverse(rb, ll);
    }
};

// 12 [3] 765 [4]
// 12 [4] 765 [3]
// 12 [4] 3567