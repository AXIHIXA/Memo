class Solution
{
public:
    int waysToMakeFair(std::vector<int> & nums)
    {
        std::ios_base::sync_with_stdio(false);
        std::cin.tie(nullptr);
        std::cout.tie(nullptr);
        
        int evenSum = 0;
        int oddSum = 0;

        for (int i = 0; i != nums.size(); ++i)
        {
            if (i & 1) oddSum += nums[i];
            else       evenSum += nums[i]; 
        }

        int ans = 0;
        bool toggle = false;

        for (int n : nums)
        {
            if (toggle)
            {
                oddSum -= n;
                ans += (oddSum == evenSum);
                evenSum += n;
            }
            else
            {
                evenSum -= n;
                ans += (oddSum == evenSum);
                oddSum += n;
            }

            toggle = !toggle;
        }

        return ans;
    }
};