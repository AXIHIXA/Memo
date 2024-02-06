class Solution
{
public:
    std::vector<std::vector<int>> combinationSum3(int k, int n)
    {
        std::vector<std::vector<int>> ans;
        std::vector<int> curr;
        backtrack(n, k, 1, 9, curr, ans);
        return ans;
    }

private:
    static void backtrack(
            int target, 
            int k, 
            int i, 
            int j, 
            std::vector<int> & curr,
            std::vector<std::vector<int>> & ans
    )
    {
        if (curr.size() == k || target <= 0)
        {
            if (target == 0 && curr.size() == k) ans.push_back(curr);
            return;
        }

        for (; i <= j; ++i)
        {
            curr.emplace_back(i);
            backtrack(target - i, k, i + 1, j, curr, ans);
            curr.pop_back();
        }
    }
};