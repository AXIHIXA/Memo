class Solution
{
public:
    std::vector<std::vector<int>> kSmallestPairs(
            std::vector<int> & nums1, 
            std::vector<int> & nums2, 
            int k
    )
    {
        auto m = static_cast<int>(nums1.size());
        auto n = static_cast<int>(nums2.size());

        std::vector<std::vector<int>> ans;

        auto cmp = [&a = nums1, &b = nums2](
                const std::pair<int, int> & p, 
                const std::pair<int, int> & q
        )
        { 
            return a[p.first] + b[p.second] > a[q.first] + b[q.second]; 
        };

        // NOTE: MUST pass in cmp as argument to heap constructor, 
        // because lambdas have no constructors 
        // (and thus could not be default-constructed). 
        std::priority_queue<
                std::pair<int, int>, 
                std::vector<std::pair<int, int>>, 
                decltype(cmp)
        > heap(cmp);  

        heap.emplace(0, 0);

        while (k-- && !heap.empty())
        {
            auto [i, j] = heap.top();
            heap.pop();

            ans.push_back({nums1[i], nums2[j]});

            if (i == 0 && j < n - 1) heap.emplace(i, j + 1);
            if (i < m - 1)           heap.emplace(i + 1, j);
        }

        return ans;
    }
};