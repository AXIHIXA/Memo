class MultiSet
{
public:
    std::vector<std::vector<int>> getSkyline(std::vector<std::vector<int>> & buildings)
    {
        auto n = static_cast<const int>(buildings.size());
        const int m = n << 1;

        using X = int;
        using Height = int;
        std::vector<std::pair<X, Height>> edges;
        edges.reserve(m);

        // 先严格按照横坐标进行「从小到大」排序
        // 对于某个横坐标而言，可能会同时出现多个点，应当按照如下规则进行处理：
        //     优先处理左端点，再处理右端点
        //     如果同样都是左端点，则按照高度「从大到小」进行处理（将高度增加到优先队列中）
        //     如果同样都是右端点，则按照高度「从小到大」进行处理（将高度从优先队列中删掉）
        for (const auto & b : buildings)
        {
            edges.emplace_back(b[0], -b[2]);
            edges.emplace_back(b[1], b[2]);
        }

        std::sort(edges.begin(), edges.end());

        // Random Element removal for heap-like interface. 
        // Heap with HashSet (lazy removal), or MULTI RBTree. 
        std::multiset<Height> tree;
        tree.emplace(0);

        std::vector<std::vector<int>> ans;

        for (int i = 0, prevHeight = 0; i < m; ++i)
        {
            auto [x, h] = edges[i];

            if (h < 0)
            {
                tree.emplace(-h);
            }
            else
            {
                // std::multiset::erase(key) erases ALL element with key `key`!
                // std::multiset::find(key) returns iterator to one element with key `key`. 
                tree.erase(tree.find(h));
            }

            int currHeight = *tree.crbegin();

            if (currHeight != prevHeight)
            {
                ans.push_back({x, currHeight});
                prevHeight = currHeight;
            }
        }

        return ans;
    }
};

class LazyRemoval
{
public:
    std::vector<std::vector<int>> getSkyline(std::vector<std::vector<int>> & buildings)
    {
        auto n = static_cast<const int>(buildings.size());
        const int m = n << 1;

        using X = int;
        using Height = int;
        std::vector<std::pair<X, Height>> edges;
        edges.reserve(m);

        // 先严格按照横坐标进行「从小到大」排序
        // 对于某个横坐标而言，可能会同时出现多个点，应当按照如下规则进行处理：
        //     优先处理左端点，再处理右端点
        //     如果同样都是左端点，则按照高度「从大到小」进行处理（将高度增加到优先队列中）
        //     如果同样都是右端点，则按照高度「从小到大」进行处理（将高度从优先队列中删掉）
        for (const auto & b : buildings)
        {
            edges.emplace_back(b[0], -b[2]);
            edges.emplace_back(b[1], b[2]);
        }

        std::sort(edges.begin(), edges.end());

        // Random Element removal for heap-like interface. 
        // Heap with HashSet (lazy removal), or MULTI RBTree.
        using TimesRemoved = int;
        std::unordered_map<Height, TimesRemoved> removed; 
        std::priority_queue<Height> maxHeap;
        maxHeap.emplace(0);

        std::vector<std::vector<int>> ans;

        for (int i = 0, prevHeight = 0; i < m; ++i)
        {
            auto [x, h] = edges[i];
            
            if (h < 0)
            {
                maxHeap.emplace(-h);
            }
            else
            {
                ++removed[h];
            }

            int currHeight = maxHeap.top();
            
            while (0 < removed[currHeight])
            {
                maxHeap.pop();
                --removed.at(currHeight);
                currHeight = maxHeap.top();
            }

            if (currHeight != prevHeight)
            {
                ans.push_back({x, currHeight});
                prevHeight = currHeight;
            }
        }

        return ans;
    }
};

using Solution = LazyRemoval;
