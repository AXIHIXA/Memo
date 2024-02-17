class Solution
{
public:
    int maxProduct(std::vector<std::string> & words)
    {
        auto n = static_cast<const int>(words.size());

        // mask[i]'s k-th bit == 1: words[i] contains 'a' + k. 
        std::vector<unsigned int> mask(n, 0U);

        for (int i = 0; i < n; ++i)
        {
            for (char c : words[i])
            {
                mask[i] |= (1 << (c - 'a'));
            }
        }

        int ans = 0;

        for (int i = 0; i < n; ++i)
        {
            for (int j = i + 1; j < n; ++j)
            {
                if (!(mask[i] & mask[j]))
                {
                    ans = std::max(ans, static_cast<int>(words[i].size() * words[j].size()));
                }
            }
        }

        return ans;
    }
};