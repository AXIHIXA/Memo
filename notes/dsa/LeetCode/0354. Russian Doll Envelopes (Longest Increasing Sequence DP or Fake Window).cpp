class Solution
{
public:
    int maxEnvelopes(std::vector<std::vector<int>> & envelopes)
    {
        int n = envelopes.size();
        if (n == 1) return 1;
        
        // Sort asecnding in width and DECREASING in height, 
        // s.t. envelopes with same length do NOT form a LIS.
        std::sort(envelopes.begin(), envelopes.end(), [](const auto & v1, const auto & v2)
        {
            return v1[0] == v2[0] ? v2[1] < v1[1] : v1[0] < v2[0];
        });

        std::vector<int> seq {envelopes[0][1]};
        seq.reserve(n);

        for (int i = 0; i < n; ++i)
        {
            if (seq.back() < envelopes[i][1])
            {
                seq.emplace_back(envelopes[i][1]);
            }
            else
            {
                auto it = std::upper_bound(
                        seq.begin(), 
                        seq.end(), 
                        envelopes[i][1], 
                        std::less_equal<int>()
                );

                // envelopes[i][1] <= seq.back() so it != seq.end()
                *it = envelopes[i][1];
            }
        }

        return seq.size();
    }

private:
    static int maxEnvelopesDP(std::vector<std::vector<int>> & envelopes)
    {
        int n = envelopes.size();
        if (n == 1) return 1;
        
        // Sort with width. Thinner envelopes in front. 
        std::sort(envelopes.begin(), envelopes.end(), [](const auto & v1, const auto & v2)
        {
            return v1[0] < v2[0];
        });

        // f[i]: Max Russian doll length using envelope i and (optionally) thinner envelopes. 
        std::vector<int> f(n);

        // g[i]: Height minimum of all known Russian doll sequences of length i. 
        // g is naturally non-decreasing. 
        std::vector<int> g(n + 1, std::numeric_limits<int>::max());
        g[0] = 0;

        // len: 1 + known max Russian doll sequence length, with width < envelopes[i]. 
        // (Note that len is used to off-the-end g, g is meaningful from index 1.)
        // When there are neighboring same-width envelopes:
        // j -> 1st in same-width group, 
        // g and len not updated (update f[i] with old g and len)
        // until i -> 1st envelop with larger width. 
        for (int i = 0, j = 0, len = 1; i < n; ++i)
        {
            if (envelopes[i][0] != envelopes[j][0])
            {
                for (; j < i; ++j)
                {
                    if (f[j] == len) g[len++] = envelopes[j][1];
                    else g[f[j]] = std::min(g[f[j]], envelopes[j][1]);
                }
            }
            
            auto it = std::upper_bound(
                    g.cbegin(), 
                    g.cbegin() + len, 
                    envelopes[i][1], 
                    std::less_equal<int>()
            );

            // it - g.cbegin() naturally >= 0. 
            f[i] = it - g.cbegin();
        }

        return *std::max_element(f.cbegin(), f.cend());
    }
};