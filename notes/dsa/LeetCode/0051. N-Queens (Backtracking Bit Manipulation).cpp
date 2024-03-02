class Solution
{
public:
    std::vector<std::vector<std::string>> solveNQueens(int n)
    {
        std::vector<int> queens;
        queens.reserve(n);
        std::vector<std::vector<std::string>> ans;
        backtrack(n, 0, 0, 0, queens, ans);
        return ans;
    }

private:
    static void backtrack(
            int n, int l, int c, int r, 
            std::vector<int> & queens, 
            std::vector<std::vector<std::string>> & ans)
    {
        if (queens.size() == n)
        {
            std::vector<std::string> tmp(n, std::string(n, '.'));
            for (int i = 0; i < n; ++i) tmp[i][queens[i]] = 'Q';
            ans.push_back(std::move(tmp));
            return;
        }

        for (int j = 0, mask = 0; j < n; ++j)
        {
            mask = (1 << j);
            if ((l | c | r) & mask) continue; 
            
            queens.emplace_back(j);
            backtrack(n, (l | mask) << 1, c | mask, (r | mask) >> 1, queens, ans);
            queens.pop_back();
        }
    }
};
