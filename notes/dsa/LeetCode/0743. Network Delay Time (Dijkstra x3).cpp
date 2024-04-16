class DijkstraDynamic
{
public:
    int networkDelayTime(std::vector<std::vector<int>> & times, int n, int k)
    {
        std::vector g(n + 1, std::vector<std::pair<int, int>>());

        for (const auto & time : times)
        {
            g[time[0]].emplace_back(time[1], time[2]);
        }

        auto cmp = [](const std::pair<int, int> & a, const std::pair<int, int> & b) -> bool
        {
            return a.second > b.second;
        };

        std::priority_queue<
                std::pair<int, int>, 
                std::vector<std::pair<int, int>>, 
                decltype(cmp)
        > heap;
        
        heap.emplace(k, 0);
        std::vector<std::uint8_t> visited(n + 1, false);
        std::vector<int> dist(n + 1, std::numeric_limits<int>::max());
        dist[k] = 0;
        
        while (!heap.empty())
        {
            auto [s, d] = heap.top();
            heap.pop();

            if (visited[s])
            {
                continue;
            }

            visited[s] = true;

            for (auto [t, w] : g[s])
            {
                if (visited[t] || dist[t] <= dist[s] + w)
                {
                    continue;
                }

                dist[t] = dist[s] + w;
                heap.emplace(t, dist[s] + w);
            }
        }

        int ans = std::numeric_limits<int>::min();

        for (int s = 1; s <= n; ++s)
        {
            if (dist[s] == std::numeric_limits<int>::max())
            {
                return -1;
            }

            ans = std::max(ans, dist[s]);
        }

        return ans;
    }
};

class DijkstraStatic
{
public:
    int networkDelayTime(std::vector<std::vector<int>> & times, int n, int k)
    {
        clear();

        for (const auto & time : times)
        {
            addEdge(time[0], time[1], time[2]);
        }

        auto cmp = [](const std::pair<int, int> & a, const std::pair<int, int> & b) -> bool
        {
            return a.second > b.second;
        };

        std::priority_queue<
                std::pair<int, int>, 
                std::vector<std::pair<int, int>>, 
                decltype(cmp)
        > heap;
        
        heap.emplace(k, 0);
        dist[k] = 0;
        
        while (!heap.empty())
        {
            auto [s, d] = heap.top();
            heap.pop();

            if (visited[s])
            {
                continue;
            }

            visited[s] = true;

            for (int e = head[s]; e != 0; e = next[e])
            {
                int t = to[e];
                int w = weight[e];
                
                if (visited[t] || dist[t] <= dist[s] + w)
                {
                    continue;
                }

                dist[t] = dist[s] + w;
                heap.emplace(t, dist[s] + w);
            }
        }

        int ans = std::numeric_limits<int>::min();

        for (int s = 1; s <= n; ++s)
        {
            if (dist[s] == std::numeric_limits<int>::max())
            {
                return -1;
            }

            ans = std::max(ans, dist[s]);
        }

        return ans;
    }

private:
    static void clear()
    {
        cnt = 0;
        std::fill(head.begin(), head.end(), 0);
        std::fill(to.begin(), to.end(), 0);
        std::fill(next.begin(), next.end(), 0);
        std::fill(weight.begin(), weight.end(), 0);

        std::fill(visited.begin(), visited.end(), false);
        std::fill(dist.begin(), dist.end(), std::numeric_limits<int>::max());
    }

    static void addEdge(int s, int t, int w)
    {
        ++cnt;
        next[cnt] = head[s];
        head[s] = cnt;
        to[cnt] = t;
        weight[cnt] = w;
    }

private:
    static constexpr int kMaxN = 110;
    static constexpr int kMaxM = 6010;

    static int cnt;
    static std::array<int, kMaxN> head;
    static std::array<int, kMaxM> to;
    static std::array<int, kMaxM> next;
    static std::array<int, kMaxM> weight;

    static std::array<bool, kMaxN> visited;
    static std::array<int, kMaxN> dist;
};

int DijkstraStatic::cnt = 0;
std::array<int, DijkstraStatic::kMaxN> DijkstraStatic::head = {0};
std::array<int, DijkstraStatic::kMaxM> DijkstraStatic::to = {0};
std::array<int, DijkstraStatic::kMaxM> DijkstraStatic::next = {0};
std::array<int, DijkstraStatic::kMaxM> DijkstraStatic::weight = {0};

std::array<bool, DijkstraStatic::kMaxN> DijkstraStatic::visited = {false};
std::array<int, DijkstraStatic::kMaxN> DijkstraStatic::dist = {};

class DijkstraBest
{
public:
    int networkDelayTime(std::vector<std::vector<int>> & times, int n, int k)
    {
        clear();

        for (const auto & time : times)
        {
            addEdge(time[0], time[1], time[2]);
        }

        heapUpdate(k, 0);
        dist[k] = 0;
        
        while (heapSize != 0)
        {
            int s = heap[0];
            popHeap();

            for (int e = head[s]; e != 0; e = next[e])
            {
                int t = to[e];
                int w = weight[e];
                heapUpdate(t, dist[s] + w);
            }
        }

        int ans = std::numeric_limits<int>::min();

        for (int s = 1; s <= n; ++s)
        {
            if (dist[s] == std::numeric_limits<int>::max())
            {
                return -1;
            }

            ans = std::max(ans, dist[s]);
        }

        return ans;
    }

private:
    static void clear()
    {
        cnt = 0;
        std::fill(head.begin(), head.end(), 0);
        std::fill(to.begin(), to.end(), 0);
        std::fill(next.begin(), next.end(), 0);
        std::fill(weight.begin(), weight.end(), 0);

        std::fill(visited.begin(), visited.end(), false);
        std::fill(dist.begin(), dist.end(), std::numeric_limits<int>::max());

        heapSize = 0;
        std::fill(heap.begin(), heap.end(), 0);
        std::fill(where.begin(), where.end(), -1);
    }

    static void addEdge(int s, int t, int w)
    {
        ++cnt;
        next[cnt] = head[s];
        head[s] = cnt;
        to[cnt] = t;
        weight[cnt] = w;
    }

    static void swap(int i, int j)
    {
        // Snapshot: heap[i] == h1, heap[j] == h2. 
        std::swap(heap[i], heap[j]);

        // Now heap[i] == h2, heap[j] == h1. 
        // Updates index as where[h2] = i, where[h1] = j. 
        where[heap[i]] = i;
        where[heap[j]] = j;
    }

    static void pushHeap(int i)
    {
        while (dist[heap[i]] < dist[heap[(i - 1) / 2]])
        {
            swap(i, (i - 1) / 2);
            i = (i - 1) / 2;
        }
    }

    static void heapify(int i)
    {
        for (int l = (i << 1) + 1; l < heapSize; )
        {
            int best = l + 1 < heapSize && dist[heap[l + 1]] < dist[heap[l]] ? l + 1 : l;
            best = dist[heap[best]] < dist[heap[i]] ? best : i;
            if (best == i) return;
            swap(best, i);
            i = best;
            l = (i << 1) + 1;
        }
    }

    static void popHeap()
    {
        swap(0, --heapSize);
        heapify(0);
        where[heap[heapSize]] = -2;
    }

    void heapUpdate(int v, int c)
    {
        if (where[v] == -1) 
        {
            heap[heapSize] = v;
            dist[v] = c;
            where[v] = heapSize++;
            pushHeap(where[v]);
        } 
        else if (0 <= where[v])
        {
            dist[v] = std::min(dist[v], c);
            pushHeap(where[v]);
        }

        // Ignore where[x] == -2 (x is popped.)
    }

private:
    static constexpr int kMaxN = 110;
    static constexpr int kMaxM = 6010;

    static int cnt;
    static std::array<int, kMaxN> head;
    static std::array<int, kMaxM> to;
    static std::array<int, kMaxM> next;
    static std::array<int, kMaxM> weight;

    static std::array<bool, kMaxN> visited;
    static std::array<int, kMaxN> dist;

    // Inverse-Indexed Heap. 
    static int heapSize;
    static std::array<int, kMaxN> heap;
    static std::array<int, kMaxN> where;
};

int DijkstraBest::cnt = 0;
std::array<int, DijkstraBest::kMaxN> DijkstraBest::head = {0};
std::array<int, DijkstraBest::kMaxM> DijkstraBest::to = {0};
std::array<int, DijkstraBest::kMaxM> DijkstraBest::next = {0};
std::array<int, DijkstraBest::kMaxM> DijkstraBest::weight = {0};

std::array<bool, DijkstraBest::kMaxN> DijkstraBest::visited = {false};
std::array<int, DijkstraBest::kMaxN> DijkstraBest::dist = {};

int DijkstraBest::heapSize = 0;
std::array<int, DijkstraBest::kMaxN> DijkstraBest::heap = {};
std::array<int, DijkstraBest::kMaxN> DijkstraBest::where = {};

// using Solution = DijkstraDynamic;
// using Solution = DijkstraStatic;
using Solution = DijkstraBest;
