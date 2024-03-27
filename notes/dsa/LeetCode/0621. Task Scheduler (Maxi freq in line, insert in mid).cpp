class Solution
{
public:
    int leastInterval(std::vector<char> & tasks, int n)
    {
        std::array<int, 26> freq = {0};

        for (char t : freq)
        {
            ++freq[t - 'A'];
        }

        int maxi = *std::max_element(freq.cbegin(), freq.cend());
        int tot = std::count(freq.cbegin(), freq.cend(), maxi);

        return std::max(static_cast<int>(tasks.size()), (n + 1) * (maxi - 1) + tot);
    }
};