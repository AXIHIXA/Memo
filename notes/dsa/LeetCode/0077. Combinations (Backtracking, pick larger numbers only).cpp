class Solution
{
public:
    std::vector<std::vector<int>> combine(int n, int k)
    {
        std::vector<std::vector<int>> ans;
        std::vector<int> comb;
        comb.reserve(k);
        
        for (int i = 1; i <= n - k + 1; ++i)
        {
            comb.emplace_back(i);
            dfs(comb, ans, n, k);
            comb.pop_back();
        }

        return ans;
    }

private:
    void dfs(std::vector<int> & comb, std::vector<std::vector<int>> & ans, int n, int k)
    {
        if (comb.size() == k)
        {
            ans.push_back(comb);
            return;
        }

        for (int x = comb.back() + 1; x <= n; ++x)
        {
            comb.emplace_back(x);
            dfs(comb, ans, n, k);
            comb.pop_back();
        }
    }
};