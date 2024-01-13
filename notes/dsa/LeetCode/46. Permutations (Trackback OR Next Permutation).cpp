class Solution 
{
public:
    std::vector<std::vector<int>> permute(std::vector<int> & nums) 
    {
        std::sort(nums.begin(), nums.end());
        std::vector<std::vector<int>> ans {nums};

        while (next_permutation(nums.begin(), nums.end()))
        {
            ans.emplace_back(nums);
        }

        return ans;
    }

private:
    template<class BidirIt>
    bool next_permutation(BidirIt first, BidirIt last)
    {
        auto r_first = std::make_reverse_iterator(last);
        auto r_last = std::make_reverse_iterator(first);
        auto left = std::is_sorted_until(r_first, r_last);
    
        if (left != r_last)
        {
            auto right = std::upper_bound(r_first, left, *left);
            std::iter_swap(left, right);
        }
    
        std::reverse(left.base(), last);
        return left != r_last;
    }

    // void backtrack(vector<int> & curr, vector<vector<int>> & ans, vector<int> & nums)
    // {
    //     if (curr.size() == nums.size())
    //     {
    //         ans.emplace_back(curr);
    //         return;
    //     }

    //     for (int n : nums)
    //     {
    //         if (find(curr.cbegin(), curr.cend(), n) == curr.cend())
    //         {
    //             curr.emplace_back(n);
    //             backtrack(curr, ans, nums);
    //             curr.pop_back();
    //         }
    //     }
    // }
};