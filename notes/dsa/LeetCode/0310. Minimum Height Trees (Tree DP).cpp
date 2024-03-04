class Solution
{
public:
    std::vector<int> findMinHeightTrees(int n, std::vector<std::vector<int>> & edges)
    {
        clear();

        for (const auto & e : edges)
        {
            add(e[0], e[1]);
            add(e[1], e[0]);
        }
        
        // Treat vertex #0 as root. 
        // f1[s]: Max distance from vertex s to its leaf children. 
        // f2[s]: "Sub"-Max distance from vertex s to its leaf children, excluding path in f1[s]. 
        // p[s]: F1-children of node s. 
        // g[s]: Max distance from vertex s to its parent, then to other leaf children (excluding s). 
        std::vector<int> f1(n);
        std::vector<int> f2(n);
        std::vector<int> p(n);
        std::vector<int> g(n);

        std::function<int (int, int)> dfs1 = [&dfs1, &f1, &f2, &p](int s, int father)
        {
            for (int e = head[s]; e != 0; e = next[e])
            {
                int t = to[e];
                if (t == father) continue;
                int sub = 1 + dfs1(t, s);

                if (f1[s] < sub)
                {
                    f2[s] = f1[s];
                    f1[s] = sub;
                    p[s] = t;
                }
                else if (f2[s] < sub)
                {
                    f2[s] = sub;
                }
            }

            return f1[s];
        };

        dfs1(0, -1);

        std::function<void (int, int)> dfs2 = [&dfs2, &f1, &f2, &p, &g](int s, int father)
        {
            for (int e = head[s]; e != 0; e = next[e])
            {
                int t = to[e];
                if (t == father) continue;
                
                if (p[s] != t) g[t] = std::max(g[t], 1 + f1[s]);
                else g[t] = std::max(g[t], 1 + f2[s]); 

                g[t] = std::max(g[t], 1 + g[s]);

                dfs2(t, s);
            }
        };

        dfs2(0, -1);

        std::vector<int> ans;

        for (int i = 0, mini = n; i < n; i++) 
        {
            int curr = std::max(f1[i], g[i]);

            if (curr < mini) 
            {
                mini = curr;
                ans.clear();
                ans.emplace_back(i);
            } 
            else if (curr == mini) 
            {
                ans.emplace_back(i);
            }
        }

        return ans;
    }

private:
    static void clear()
    {
        cnt = 1;
        std::fill(head.begin(), head.end(), 0);
        std::fill(to.begin(), to.end(), 0);
        std::fill(next.begin(), next.end(), 0);
    }

    static void add(int s, int t)
    {
        next[cnt] = head[s];
        head[s] = cnt;
        to[cnt] = t;
        ++cnt;
    }

private:
    // 1 <= n <= 2 * 1e4
    static constexpr int N = 20'010;
    static constexpr int M = N << 1;

    // Static Adjancency List (Foward-Star List). 
    // 0 means this node has no edges associated. 
    static int cnt;                  // Edge count, start from 1. 
    static std::array<int, N> head;  // Vertex attribute, to 1st edge leaving this vertex. 
    static std::array<int, M> to;    // Edge attribute, target of this edge. 
    static std::array<int, M> next;  // Edge attribute, to next edge having the same source vertex. 
};

int Solution::cnt = 1;
std::array<int, Solution::N> Solution::head;
std::array<int, Solution::M> Solution::to;
std::array<int, Solution::M> Solution::next;
