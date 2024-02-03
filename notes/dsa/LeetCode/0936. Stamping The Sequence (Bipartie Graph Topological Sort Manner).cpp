class Solution
{
public:
    std::vector<int> movesToStamp(std::string stamp, std::string target)
    {
        int m = stamp.size();
        int n = target.size();

        // Build a bi-partie graph. 
        // Left vertices: Locations in target;
        // Right vertices: Stampable locations in target.
        // An edge (l -> r) added iff. stamping at r incurrs wrong char at location l. 
        std::vector<int> inDegree(n - m + 1, m);
        std::vector<std::vector<int>> al(n, std::vector<int>());
        
        std::queue<int> qu;
        
        for (int i = 0; i <= n - m; i++) 
        {
			for (int j = 0; j < m; j++) 
            {
				if (target[i + j] == stamp[j])
                {
                    if (--inDegree[i] == 0)
                    {
                        qu.emplace(i);
                    }
				} 
                else 
                {
					// Stamping at i incurrs wrong char at location i + j. 
                    al[i + j].emplace_back(i);
				}
			}
		}

        std::vector<int> ans;
        
        // 同一个位置取消错误不要重复统计
        std::vector<bool> visited(n, false);

        while (!qu.empty())
        {
            int curr = qu.front();
            qu.pop();
            ans.emplace_back(curr);
            
            for (int i = 0; i < m; ++i)
            {
                if (!visited[curr + i])
                {
                    visited[curr + i] = true;

                    for (int next : al[curr + i])
                    {
                        if (--inDegree[next] == 0)
                        {
                            qu.emplace(next);
                        }
                    }
                }
            }
        }

        if (ans.size() != n - m + 1) return {};
        std::reverse(ans.begin(), ans.end());
        return ans;
    }
};