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
        // OR go this syntax (with C++17 deduction guide): 
        // std::priority_queue heap(cmp, std::vector<std::pair<int, int>> {});  
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

            // Avoid duplicates by extending j iff. current smallest i == 0. 
            // Proof of correctness:
            // Current pair i, j, i != 0. 
            // Does not include i, j + 1 because it's either not the next optimal, 
            // or in heap already. 
            // Since we have i, j, in heap, 
            // we must have 0, j in heap previously (which is now popped already). 
            // Upon its popping, 0, j + 1 is pushed into heap. 
            // If 0, j + 1 is still in heap, then i, j + 1 is NOT the next optimal. 
            // If 0, j + 1 is popped, then 1, j + 1 should be in heap, ...
            // Thus i, j + 1 is either NOT the next optimal or already in heap. 
            if (i == 0 && j < n - 1) heap.emplace(i, j + 1);
            if (i < m - 1)           heap.emplace(i + 1, j);
        }

        return ans;
    }
};