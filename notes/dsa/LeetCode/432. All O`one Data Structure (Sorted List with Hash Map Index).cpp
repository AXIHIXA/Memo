class AllOne 
{
public:
    AllOne()
    {
        ios_base::sync_with_stdio(false);
        cin.tie(nullptr);
        cout.tie(nullptr);
    }
    
    void inc(string key)
    {
        // Ensure a dummy (or real) node with key
        if (idx.find(key) == idx.end())
        {
            idx[key] = lst.insert(lst.begin(), {0, {key}});
        }
        
        // Insert key into the next node
        // (existing node or a new node inserted before the next node)
        auto bktItNext = idx[key], bktIt = bktItNext++;

        if (bktItNext == lst.end() or bktIt->val + 1 < bktItNext->val)
        {
            bktItNext = lst.insert(bktItNext, {bktIt->val + 1, {}});
        }

        bktItNext->keys.insert(key);
        idx[key] = bktItNext;

        // Erase the old node (real or dummy)
        bktIt->keys.erase(key);

        if (bktIt->keys.empty())
        {
            lst.erase(bktIt);
        }
    }
    
    void dec(string key)
    {
        if (auto it = idx.find(key); it == idx.end())
        {
            // Existance of key ensured by problem statement; 
            // for completeness only. 
            return;
        }
        else
        {
            // Insert into the prev node
            // (existing node or a newly-inserted one)
            auto bktItPrev = it->second, bktIt = bktItPrev--;
            idx.erase(key);

            if (1 < bktIt->val)
            {
                if (bktIt == lst.begin() or bktItPrev->val + 1 < bktIt->val)
                {
                    bktItPrev = lst.insert(bktIt, {bktIt->val - 1, {}});
                }
                
                bktItPrev->keys.insert(key);
                idx[key] = bktItPrev;
            }

            // Erase the old node if empty
            bktIt->keys.erase(key);

            if (bktIt->keys.empty())
            {
                lst.erase(bktIt);
            }
        }
    }
    
    string getMaxKey()
    {
        return lst.empty() ? "" : *(lst.back().keys.begin());
    }
    
    string getMinKey()
    {
        return lst.empty() ? "" : *(lst.front().keys.begin());
    }

private:
    struct Bucket
    {
        Bucket() = default;
        Bucket(int v, std::initializer_list<std::string> l) : val(v), keys(std::move(l)) {}
        
        int val {0};
        std::unordered_set<std::string> keys {};
    };

    using BucketList = std::list<Bucket>;

    BucketList lst {};
    std::unordered_map<std::string, BucketList::iterator> idx {};
};


/**
 * Your AllOne object will be instantiated and called as such:
 * AllOne* obj = new AllOne();
 * obj->inc(key);
 * obj->dec(key);
 * string param_3 = obj->getMaxKey();
 * string param_4 = obj->getMinKey();
 */