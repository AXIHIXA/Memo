class AllOne
{
public:
    AllOne() = default;
    
    void inc(std::string key)
    {
        auto idxIt = idx.find(key);
        
        if (idxIt == idx.end())
        {
            if (lst.empty() || lst.front().count != 1)
            {
                lst.emplace_front(key);
            }
            else
            {
                lst.front().data.emplace(key);
            }

            idx.emplace(key, lst.begin());
        }
        else
        {
            auto currNode = idxIt->second;
            auto targetNode = std::next(currNode);

            if (targetNode == lst.end() || targetNode->count != currNode->count + 1)
            {
                targetNode = lst.emplace(targetNode, currNode->count + 1, key);
            }
            else
            {
                targetNode->data.emplace(key);
            }

            idx.at(key) = targetNode;
            currNode->data.erase(key);
            if (currNode->data.empty()) lst.erase(currNode);
        }
    }
    
    void dec(std::string key)
    {
        // It is guaranteed that key exists in the data structure before the decrement.
        auto currNode = idx.at(key);
        
        if (currNode->count == 1)
        {
            idx.erase(key);
            currNode->data.erase(key);
            if (currNode->data.empty()) lst.erase(currNode);
        }
        else
        {
            auto targetNode = std::prev(currNode);
            
            if (currNode == lst.begin() || targetNode->count != currNode->count - 1)
            {
                targetNode = lst.emplace(currNode, currNode->count - 1, key);
            }
            else
            {
                targetNode->data.emplace(key);
            }

            idx.at(key) = targetNode;
            currNode->data.erase(key);
            if (currNode->data.empty()) lst.erase(currNode);
        }
    }
    
    std::string getMaxKey()
    {
        return lst.empty() ? "" : *lst.back().data.begin();
    }
    
    std::string getMinKey()
    {
        return lst.empty() ? "" : *lst.front().data.begin();
    }

private:
    struct Bucket
    {
        explicit Bucket(std::string key) : count(1), data {key} {}
        Bucket(int count, std::string key) : count(count), data {key} {}
        
        int count = 0;
        std::unordered_set<std::string> data;
    };

    using List = std::list<Bucket>;
    using ListIter = List::iterator;

    List lst;
    std::unordered_map<std::string, ListIter> idx;
};

/**
 * Your AllOne object will be instantiated and called as such:
 * AllOne* obj = new AllOne();
 * obj->inc(key);
 * obj->dec(key);
 * string param_3 = obj->getMaxKey();
 * string param_4 = obj->getMinKey();
 */