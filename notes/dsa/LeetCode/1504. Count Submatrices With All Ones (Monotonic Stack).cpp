class Solution
{
public:
    int numSubmat(std::vector<std::vector<int>> & mat)
    {
        auto m = static_cast<const int>(mat.size());
        auto n = static_cast<const int>(mat.front().size());
        
        std::vector<int> h = mat[0];

        std::vector<int> stk;
        stk.reserve(n);

        int ans = 0;

        for (int row = 0; row < m; ++row)
        {
            if (0 < row)
            {
                for (int col = 0; col < n; ++col)
                {
                    h[col] = mat[row][col] == 1 ? h[col] + 1 : 0;
                }
            }

            stk.clear();

            for (int i = 0; i < n; ++i)
            {
                while (!stk.empty() && h[i] <= h[stk.back()])
                {
                    int cur = stk.back();
                    stk.pop_back();

                    if (h[cur] == 0)
                    {
                        continue;
                    }

                    int ll = stk.empty() ? -1 : stk.back();
                    int rr = i;
                    int len = rr - ll - 1;
                    int bottom = std::max((ll == -1 ? 0 : h[ll]), h[rr]);

                    // These number of matrices: 
                    // (1) Bottomed at line row;
                    // (2) Must contain column cur;
                    // (3) With height s.t. h[ll] < h <= h[cur];
                    ans += (h[cur] - bottom) * len * (len + 1) / 2;
                }

                stk.emplace_back(i);
            }

            while (!stk.empty())
            {
                int cur = stk.back();
                stk.pop_back();
                
                if (h[cur] == 0)
                {
                    continue;
                }

                int ll = stk.empty() ? -1 : stk.back();
                int rr = n;
                int len = rr - ll - 1;
                int bottom = std::max((ll == -1 ? 0 : h[ll]), 0);

                ans += (h[cur] - bottom) * len * (len + 1) / 2;
            }
        }

        return ans;
    }
};