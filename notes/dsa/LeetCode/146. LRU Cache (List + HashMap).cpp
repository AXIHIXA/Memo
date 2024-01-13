class LRUCache 
{
public:
    LRUCache(int capacity) : capacity(capacity)
    {
        
    }
    
    int get(int key) 
    {
        if (auto it = index.find(key); it == index.end())
        {
            return -1;
        }
        else
        {
            // Already in cache. 

            // Update and move entry to mru. 
            auto lruIter = it->second;
            int value = lruIter->second;
            lru.erase(lruIter);
            lru.emplace_back(key, value);

            // Update index. 
            it->second = --lru.end();

            return value;
        }
    }
    
    void put(int key, int value) 
    {
        if (auto it = index.find(key); it == index.end())
        {
            // Not in cache. 

            if (lru.size() == capacity)
            {
                // Remove lru. 
                index.erase(lru.begin()->first);
                lru.erase(lru.begin());
            }
            
            // Insert. 
            lru.emplace_back(key, value);
            index[key] = --lru.end();
        }
        else
        {
            // Already in cache. 

            // Update and move entry to mru. 
            auto lruIter = it->second;
            lru.erase(lruIter);
            lru.emplace_back(key, value);

            // Update index. 
            it->second = --lru.end();
        }
    }

private:
    int capacity {0};
    list<pair<int, int>> lru;
    unordered_map<int, list<pair<int, int>>::iterator> index;
};

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache* obj = new LRUCache(capacity);
 * int param_1 = obj->get(key);
 * obj->put(key,value);
 */