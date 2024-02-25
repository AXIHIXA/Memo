class Solution
{
public:
    std::vector<int> deckRevealedIncreasing(std::vector<int> & deck)
    {
        auto n = static_cast<const int>(deck.size());
        std::sort(deck.begin(), deck.end());

        std::vector<int> ans(n);
        
        // Put indices of ans into qu. 
        std::queue<int> qu;
        for (int i = 0; i < n; ++i) qu.emplace(i);

        // Reverse simulation + filling using sorted deck. 
        for (int i = 0; i < n; ++i)
        {
            // 1. Reveal. 
            ans[qu.front()] = deck[i];
            qu.pop();

            // 2. Push front to bottom. 
            if (!qu.empty())
            {
                int x = qu.front();
                qu.pop();
                qu.push(x);
            }
        }

        return ans;
    }
};