class Solution
{
public:
    int halveArray(std::vector<int> & nums)
    {
        double sum = std::accumulate(nums.cbegin(), nums.cend(), 0.0);
        double half_sum = sum * 0.5;

        std::priority_queue<double> heap;
        for (int x : nums) heap.emplace(x);

        int ans = 0;

        while (half_sum < sum)
        {
            double x = heap.top();
            heap.pop();

            double half_x = x * 0.5;
            sum -= half_x;
            heap.emplace(half_x);

            ++ans;
        }

        return ans;
    }
};
