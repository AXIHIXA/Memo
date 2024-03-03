class Solution
{
public:
    std::string nextPalindrome(std::string num)
    {
        auto m = static_cast<const int>(num.size());
        const int n = (m >> 1);

        if (!nextPermutation(num.begin(), num.begin() + n))
        {
            return "";
        }

        for (int i = 0; i < n; ++i)
        {
            num[m - 1 - i] = num[i];
        }

        return num;
    }

private:
    template <typename Iter>
    static bool nextPermutation(Iter b, Iter e)
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

        return ll != re;
    }

    // 12 3 765 4
    // 12 4 765 3
    // 12 4 3567

    // 7654321

    // 01234
};