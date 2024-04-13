class DP
{
public:
    int minStickers(std::vector<std::string> & stickers, std::string target)
    {
        auto n = static_cast<const int>(target.size());
        
        // dp[s]: #Stickers needed to fill target's indices as in s's set-bits. 
        const int m = 1 << n;
        std::vector<int> dp(m, -1);
        dp[0] = 0;

        for (int s = 0; s < m; ++s)
        {
            if (dp[s] == -1)
            {
                continue;
            }

            for (const std::string & str : stickers)
            {
                int t = s;

                for (char c : str)
                {
                    for (int i = 0; i < n; ++i)
                    {
                        if (((t >> i) & 1) == 1)
                        {
                            continue;
                        }

                        if (c == target[i])
                        {
                            t |= (1 << i);
                            break;
                        }
                    }
                }

                if (dp[t] == -1 || 1 + dp[s] < dp[t])
                {
                    dp[t] = 1 + dp[s];
                }
            }
        }

        return dp.back();
    }
};

class BFS
{
public:
    int minStickers(std::vector<std::string> & stickers, std::string target)
    {   
        for (auto & s : stickers)
        {
            std::sort(s.begin(), s.end());
        }

        std::sort(target.begin(), target.end());
        
        auto n = static_cast<const int>(target.size());
        const int m = 1 << n;
        const int tm = m - 1;

        // BFS target into empty string. 
        // Use binary mask of target's indices resolved. 
        std::queue<int> que;
        que.emplace(tm);

        std::vector<std::uint8_t> visited(m, false);
        visited[tm] = true;

        int level = 0;

        while (!que.empty())
        {
            auto levelSize = static_cast<const int>(que.size());
            ++level;

            for (int i = 0; i < levelSize; ++i)
            {
                int cur = que.front();
                que.pop();

                for (const auto & s : stickers)
                {
                    int dur = cur;
                    
                    for (int i = 0, j = 0; i < n && j < static_cast<const int>(s.size()); )
                    {
                        if (((dur >> i) & 1) == 0)
                        {
                            ++i;
                            continue;
                        }

                        if (target[i] < s[j])
                        {
                            ++i;
                        }
                        else if (s[j] < target[i])
                        {
                            ++j;
                        }
                        else
                        {
                            dur ^= (1 << i);
                            ++i;
                            ++j;
                        }
                    }

                    // std::bitset<16> bs1(cur);
                    // std::bitset<16> bs2(dur);
                    // std::printf(
                    //     "%d: %s -- %s --> %s\n", 
                    //     level, 
                    //     bs1.to_string().c_str(), 
                    //     s.c_str(), 
                    //     bs2.to_string().c_str()
                    // );

                    if (dur == 0)
                    {
                        return level;
                    }
                    else if (!visited[dur])
                    {
                        visited[dur] = true;
                        que.emplace(dur);
                    }
                }
            }
        }

        return -1;
    }
};

// using Solution = DP;
using Solution = BFS;
