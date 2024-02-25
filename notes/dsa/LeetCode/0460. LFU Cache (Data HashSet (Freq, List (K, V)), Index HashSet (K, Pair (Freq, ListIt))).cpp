class LFUCache
{
public:
    using Freq = int;
    using K = int;
    using V = int;
    using EntryList = std::list<std::pair<K, V>>;
    using EntryListIter = EntryList::iterator;

public:
    LFUCache(int capacity) : capacity(capacity) {}
    
    int get(K key)
    {
        auto idxIt = index.find(key);
        if (idxIt == index.end()) return -1;

        auto [freq, entryListIt] = idxIt->second;
        auto val = entryListIt->second;

        erase(key);
        insert(key, val, freq + 1);

        return val;
    }
    
    void put(K key, V val)
    {
        auto idxIt = index.find(key);

        if (idxIt != index.end())
        {
            // Note: In this case freq is old + 1, not 1!!!
            idxIt->second.second->second = val;
            get(key);
            return;
        }
        
        if (capacity == index.size())
        {
            K lfuKey = data.at(minFreq).front().first;
            erase(lfuKey);
        }
        
        insert(key, val, 1);
        minFreq = 1;
    }

private:
    void insert(K key, V val, Freq freq)
    {
        // Key MUST NOT exist. 
        // Does NOT update minFreq. 
        data[freq].emplace_back(key, val);
        // index.emplace(key, std::make_pair(freq, std::prev(data.at(freq).end())));
        index[key] = std::make_pair(freq, std::prev(data.at(freq).end()));
    }

    void erase(K key)
    {
        // Key MUST exist. 
        // Updates minFreq. 
        auto [freq, entryListIt] = index.at(key);
        data.at(freq).erase(entryListIt);

        if (data.at(freq).empty())
        {
            data.erase(freq);
            if (minFreq == freq) ++minFreq;
        }

        index.erase(key);
    }

private:
    int capacity = 0;
    int minFreq = 1;
    std::unordered_map<Freq, EntryList> data;
    std::unordered_map<K, std::pair<Freq, EntryListIter>> index;
};

/**
 * Your LFUCache object will be instantiated and called as such:
 * LFUCache* obj = new LFUCache(capacity);
 * int param_1 = obj->get(key);
 * obj->put(key,value);
 */