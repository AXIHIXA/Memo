class Solution 
{
public:
    // Suppose we have array:
    //     j elements of value != me
    //     >= j + 1 elements of value me. 
    // If we remove one me and one (!=me) from this array, 
    // me remains to be the majority element. 
    // So this algirithm is essentially
    // Finding these pairs and removing them. 
    // When no such pair could be found, 
    // all elements remaining will be me. 
    int majorityElement(std::vector<int> & nums) 
    {
        int count = 0;
        int ans = nums[0];

        for (int e : nums)
        {
            if (count == 0) ans = e;
            count += (e == ans) ? 1 : -1;
        }

        return ans;
    }
};