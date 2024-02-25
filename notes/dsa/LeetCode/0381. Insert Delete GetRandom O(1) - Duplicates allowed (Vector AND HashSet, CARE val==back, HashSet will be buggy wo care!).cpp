class RandomizedCollection
{
public:
    RandomizedCollection() = default;
    
    bool insert(int val)
    {
        bool ret = !(index.contains(val));
        data.emplace_back(val);
        index[val].emplace(data.size() - 1);
        return ret;
    }
    
    bool remove(int val)
    {
        auto it = index.find(val);
        if (it == index.end()) return false;

        std::unordered_set<int> & valIds = it->second;
        int i = *valIds.begin();
        int bak = data.back();
        int j = data.size() - 1;

        // Special care MUST be taken for identical val/bak values!
        // Doing push/pops on the same index HashSet will be buggy!
        if (val == bak)
        {
            valIds.erase(j);
        }
        else
        {
            std::unordered_set<int> & bakIds = index.at(bak);
            bakIds.emplace(i);
            data[i] = bak;
            bakIds.erase(j);
            valIds.erase(i);
        }

        data.pop_back();

        if (valIds.empty()) index.erase(it);

        return true;
    }
    
    int getRandom()
    {
        return data[std::rand() % data.size()];
    }

private:
    std::vector<int> data;

    // {Value, Indices in data}
    std::unordered_map<int, std::unordered_set<int>> index;
};

/**
 * Your RandomizedCollection object will be instantiated and called as such:
 * RandomizedCollection* obj = new RandomizedCollection();
 * bool param_1 = obj->insert(val);
 * bool param_2 = obj->remove(val);
 * int param_3 = obj->getRandom();
 */