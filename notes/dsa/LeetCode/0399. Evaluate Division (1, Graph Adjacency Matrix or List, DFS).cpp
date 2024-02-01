class Solution
{
public:
    std::vector<double> calcEquation(
            std::vector<std::vector<std::string>> & equations, 
            std::vector<double> & values, 
            std::vector<std::vector<std::string>> & queries
    )
    {
        int k = 0;
        std::unordered_map<std::string, int> sid;

        for (const auto & e : equations)
        {
            for (const std::string & s : e)
            {
                if (sid.find(s) == sid.end())
                {
                    sid.emplace(s, k++);
                }
            }
        }

        std::vector<std::vector<double>> am(k, std::vector<double>(k, 0.0));

        for (int i = 0, es = equations.size(); i < es; ++i)
        {
            int s = sid.at(equations[i][0]);
            int t = sid.at(equations[i][1]);
            am[s][t] = values[i];
            am[t][s] = 1.0 / values[i];
        }

        std::vector<double> ans;
        ans.reserve(queries.size());

        for (const auto & q : queries)
        {
            auto it = sid.find(q[0]);
            auto jt = sid.find(q[1]);

            if (it == sid.end() || jt == sid.end())
            {
                ans.emplace_back(-1.0);
                continue;
            }

            int s = it->second;
            int t = jt->second;

            std::stack<std::pair<int, double>> st;
            std::vector<bool> visited(k, false);
            st.emplace(s, 1.0);

            [&st, &visited, &am, &ans, t, k] 
            {
                while (!st.empty())
                {
                    auto [curr, res] = st.top();
                    st.pop();
                    visited[curr] = true;

                    if (curr == t)
                    {
                        ans.emplace_back(res);
                        return;
                    }

                    for (int next = 0; next < k; ++next)
                    {
                        if (!visited[next] && am[curr][next] != 0.0)
                        {
                            st.emplace(next, res * am[curr][next]);
                        }
                    }
                }

                ans.emplace_back(-1.0);
            }();
        }

        return ans;
    }
};