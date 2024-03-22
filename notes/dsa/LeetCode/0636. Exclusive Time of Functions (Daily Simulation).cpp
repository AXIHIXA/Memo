class Solution
{
public:
    std::vector<int> exclusiveTime(int n, std::vector<std::string> & logs)
    {
        using Id = int;
        using StartTime = int;
        std::stack<std::pair<Id, StartTime>> stk;

        std::vector<int> ans(n, 0);

        for (const auto & log : logs)
        {
            std::size_t c1 = log.find_first_of(':', 0);
            std::size_t c2 = log.find_first_of(':', c1 + 1);
            int funcId = std::stoi(log.substr(0, c1));
            int currTime = std::stoi(log.substr(c2 + 1));
            
            if (log[c1 + 1] == 's')
            {
                // start
                if (!stk.empty())
                {
                    auto [id, startTime] = stk.top();
                    ans[id] += currTime - startTime;
                }

                stk.emplace(funcId, currTime);
            }
            else
            {
                // end
                ans[funcId] += currTime - stk.top().second + 1;
                stk.pop();
                
                if (!stk.empty())
                {
                    stk.top().second = currTime + 1;
                }
            }
        }

        return ans;
    }
};