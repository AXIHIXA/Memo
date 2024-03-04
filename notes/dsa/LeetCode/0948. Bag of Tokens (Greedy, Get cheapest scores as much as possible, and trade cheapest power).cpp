class Solution
{
public:
    int bagOfTokensScore(std::vector<int> & tokens, int power)
    {
        auto n = static_cast<const int>(tokens.size());
        std::sort(tokens.begin(), tokens.end());

        int score = 0;
        int ans = 0;

        // While we could still play iff. both hold:
        // (1) Have cards unplayed;
        // (2) Could play either face-up or face-down. 
        for (int lo = 0, hi = n - 1; lo <= hi && (tokens[lo] <= power || 0 < score); )
        {
            // Play face-up as much as possible. 
            while (lo <= hi && tokens[lo] <= power)
            {
                power -= tokens[lo++];
                ++score;
            }

            ans = std::max(ans, score);

            // Trade the cheapest power with 1 score point. 
            if (lo <= hi && 0 < score)
            {
                power += tokens[hi--];
                --score;
            }
        }

        return ans;
    }
};