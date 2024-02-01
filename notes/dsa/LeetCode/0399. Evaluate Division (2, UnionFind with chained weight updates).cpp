class UnionFind
{
public:
    UnionFind(
            const std::vector<std::vector<std::string>> & equations, 
            const std::vector<double> & values 
    )
    {
        int k = 0;  // k symbols in total. 

        for (int i = 0, es = equations.size(); i < es; ++i)
        {
            const std::string & a = equations[i][0];
            const std::string & b = equations[i][1];

            if (mp.find(a) == mp.end()) mp.emplace(a, k++);
            if (mp.find(b) == mp.end()) mp.emplace(b, k++);
        }

        root.assign(k, 0);
        std::iota(root.begin(), root.end(), 0);
        rank.assign(k, 0);
        weight.assign(k, std::vector<double>(k, 1.0));

        for (int i = 0, es = equations.size(); i < es; ++i)
        {
            const std::string & a = equations[i][0];
            const std::string & b = equations[i][1];
            double q = values[i];

            int x = mp.at(a);
            int y = mp.at(b);

            merge(x, y, q);
        }
    }

    int find(int x)
    {
        int rx = root[x];

        if (rx == x) return x;
        
        root[x] = find(rx);

        if (root[x] != rx)
        {
            weight[x][root[x]] = weight[x][rx] * weight[rx][root[x]];
            weight[root[x]][x] = 1.0 / weight[x][root[x]];
        }

        return root[x];
    }

    void merge(int x, int y, double q)
    {
        int rx = find(x);
        int ry = find(y);

        weight[x][y] = q;
        weight[y][x] = 1.0 / q;

        if (rx == ry) return;

        weight[rx][ry] = weight[rx][x] * weight[y][ry] * weight[x][y];
        weight[ry][rx] = 1.0 / weight[rx][ry];
        
        if (rank[rx] < rank[ry]) root[rx] = ry;
        else if (rank[ry] < rank[rx]) root[ry] = root[rx];
        else root[ry] = rx, ++rank[rx];
    }

    std::unordered_map<std::string, int> mp;
    
    std::vector<int> root;
    std::vector<int> rank;

    // a / b == q, weight[a][b] <- q, weight[b][a] <- 1/q. 
    std::vector<std::vector<double>> weight;
};

class Solution
{
public:
    std::vector<double> calcEquation(
            std::vector<std::vector<std::string>> & equations, 
            std::vector<double> & values, 
            std::vector<std::vector<std::string>> & queries)
    {
        UnionFind uf {equations, values};

        std::vector<double> ans;
        ans.reserve(queries.size());

        for (const auto & q : queries)
        {
            const std::string & a = q[0];
            const std::string & b = q[1];
            
            auto it = uf.mp.find(a);
            auto jt = uf.mp.find(b);

            if (it == uf.mp.end() || jt == uf.mp.end())
            {
                ans.emplace_back(-1.0);
                continue;
            }

            int x = it->second;
            int y = jt->second;

            int rx = uf.find(x);
            int ry = uf.find(y);

            if (rx != ry)
            {
                ans.emplace_back(-1.0);
                continue;
            }

            ans.emplace_back(uf.weight[x][rx] * uf.weight[rx][y]);
        }

        return ans;
    }
};
