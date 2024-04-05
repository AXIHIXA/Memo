class Solution
{
public:
    int maximumInvitations(std::vector<int> & favorite)
    {
        auto n = static_cast<const int>(favorite.size());
        
        std::vector<int> inDegree(n, 0);

        for (int s = 0; s < n; ++s)
        {
            ++inDegree[favorite[s]];
        }

        std::queue<int> que;

        for (int v = 0; v < n; ++v)
        {
            if (inDegree[v] == 0)
            {
                que.emplace(v);
            }
        }

        // deep[i]: 不包括i在内，i之前的最长链的长度
        std::vector<int> deep(n, 0);

        while (!que.empty())
        {
            int s = que.front();
            que.pop();

            int t = favorite[s];
            deep[t] = std::max(deep[t], deep[s] + 1);

            if (--inDegree[t] == 0)
            {
                que.emplace(t);
            }
        }

        // 目前图中的点，不在环上的点，都删除了！ indegree[i] == 0
		// 可能性1 : 所有小环(中心个数 == 2)，算上中心点 + 延伸点，总个数
		int sumOfSmallRings = 0;
		// 可能性2 : 所有大环(中心个数 > 2)，只算中心点，最大环的中心点个数
        int sizeOfBiggestRing = 0;

        for (int s = 0; s < n; ++s)
        {
            if (0 < inDegree[s])
            {
				inDegree[s] = 0;
                
                int t = favorite[s];
                inDegree[t] = 0;

                if (favorite[t] == s)
                {
                    // Small ring
                    sumOfSmallRings += 2 + deep[s] + deep[t];
                }
                else
                {
                    // Large ring
                    int sizeOfThisRing = 1;

                    for ( ; t != s; t = favorite[t])
                    {
                        ++sizeOfThisRing;
                        inDegree[t] = 0;
                    }

                    sizeOfBiggestRing = std::max(sizeOfBiggestRing, sizeOfThisRing);
                }
            }
        }

        return std::max(sumOfSmallRings, sizeOfBiggestRing);
    }
};