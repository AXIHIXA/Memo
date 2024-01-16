class RandomizedSet 
{
public:
    RandomizedSet() = default;

    bool insert(int val) 
    {
        if (idx.find(val) != idx.end()) return false;
        idx.emplace(val, data.size());
        data.emplace_back(val);
        return true;
    }
    
    bool remove(int val) 
    {
        auto it = idx.find(val);
        if (it == idx.end()) return false;

        data[it->second] = data.back();
        idx.at(data.back()) = it->second;
        
        data.pop_back();
        idx.erase(it);

        return true;
    }
    
    int getRandom() 
    {
        return data[std::rand() % data.size()];
    }

private:
    std::unordered_map<int, int> idx;
    std::vector<int> data;
};

/**
 * Your RandomizedSet object will be instantiated and called as such:
 * RandomizedSet* obj = new RandomizedSet();
 * bool param_1 = obj->insert(val);
 * bool param_2 = obj->remove(val);
 * int param_3 = obj->getRandom();
 */