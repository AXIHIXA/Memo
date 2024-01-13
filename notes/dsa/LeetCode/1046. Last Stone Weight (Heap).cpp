class Solution 
{
public:
    int lastStoneWeight(std::vector<int> & stones) 
    {
        std::priority_queue q(stones.cbegin(), stones.cend());

        while (!q.empty())
        {
            int sz = q.size();

            if (sz == 1)
            {
                return q.top();
            }
            else 
            {
                int s1 = q.top();
                q.pop();

                int s2 = q.top();
                q.pop();

                if (s1 != s2)
                {
                    q.push(std::abs(s1 - s2));
                }
                else
                {
                    if (q.size() == 0)
                    {
                        return 0;
                    }
                }
            }
        }

        return q.top();
    }
};