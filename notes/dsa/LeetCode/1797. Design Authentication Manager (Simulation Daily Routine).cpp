class AuthenticationManager
{
public:
    AuthenticationManager(int timeToLive) : timeToLive(timeToLive)
    {
        
    }
    
    void generate(std::string tokenId, int currentTime)
    {
        hashMap[tokenId] = currentTime + timeToLive;
    }
    
    void renew(std::string tokenId, int currentTime)
    {
        auto it = hashMap.find(tokenId);

        if (it == hashMap.end())
        {
            return;
        }

        int & expireTime = it->second;

        if (currentTime < expireTime)
        {
            expireTime = currentTime + timeToLive;
        }
        else
        {
            hashMap.erase(it);
        }
    }
    
    int countUnexpiredTokens(int currentTime)
    {
        int ans = 0;

        std::vector<std::string> expiredTokens;
        
        for (auto & [tokenId, expireTime] : hashMap)
        {
            if (currentTime < expireTime)
            {
                ++ans;
            }
            else
            {
                expiredTokens.emplace_back(tokenId);
            }
        }

        for (const auto & tokenId : expiredTokens)
        {
            hashMap.erase(tokenId);
        }

        return ans;
    }

private:
    std::unordered_map<std::string, int> hashMap;

    const int timeToLive;
};

/**
 * Your AuthenticationManager object will be instantiated and called as such:
 * AuthenticationManager* obj = new AuthenticationManager(timeToLive);
 * obj->generate(tokenId,currentTime);
 * obj->renew(tokenId,currentTime);
 * int param_3 = obj->countUnexpiredTokens(currentTime);
 */
 */