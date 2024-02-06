class Solution
{
public:
    std::vector<std::vector<int>> combinationSum(std::vector<int> & candidates, int target)
    {
        std::vector<std::vector<int>> ans;
        std::vector<int> curr;
        dfs(candidates, 0, target, curr, ans);
        return ans;
    }

private:
    // Use ONLY candidates[i:]. 
    static void dfs(
            const std::vector<int> & candidates, 
            int i, 
            int target, 
            std::vector<int> & curr,
            std::vector<std::vector<int>> & ans
    )
    {
        if (target <= 0)
        {
            if (target == 0) ans.push_back(curr);
            return;
        }

        for (; i < candidates.size(); ++i)
        {
            curr.emplace_back(candidates[i]);
            dfs(candidates, i, target - candidates[i], curr, ans);
            curr.pop_back();
        }
    }
};