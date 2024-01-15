class Solution 
{
public:
    Solution(std::vector<int> & nums) : arr(nums), ori(nums)
    {
        std::ios_base::sync_with_stdio(false);
        std::cin.tie(nullptr);
        std::cout.tie(nullptr);
    }
    
    std::vector<int> reset() 
    {
        return arr = ori;
    }
    
    std::vector<int> shuffle() 
    {
        // Fisher-Yates Algorithm. 
        // For each element, swap with a uniform rand successor element. 
        for (int i = 0; i != arr.size(); ++i)
        {
            std::swap(arr[i], arr[i + std::rand() % (arr.size() - i)]);
        }

        return arr;
    }

private:
    std::vector<int> & arr;
    std::vector<int> ori;
};

/**
 * Your Solution object will be instantiated and called as such:
 * Solution* obj = new Solution(nums);
 * vector<int> param_1 = obj->reset();
 * vector<int> param_2 = obj->shuffle();
 */