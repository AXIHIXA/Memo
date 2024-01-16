class Solution 
{
public:
    int hIndex(vector<int> & citations) 
    {
        int n = citations.size();
        std::vector<int> count(n + 1, 0);
        for (int c : citations) ++count[std::min(c, n)];
        int ans = n;
        for (int s = count[n]; s < ans; s += count[ans]) --ans;
        return ans;
    }
};