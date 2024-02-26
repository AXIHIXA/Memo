class HashSetAndHeap
{
public:
    HashSetAndHeap() = default;
    
    int popSmallest()
    {
        int ans;

        if (!added.empty())
        {
            ans = added.top();
            added.pop();
            isPresent.erase(ans);
        }
        else
        {
            ans = curr++;
        }

        return ans;
    }
    
    void addBack(int num)
    {
        if (curr <= num || isPresent.contains(num)) return;
        added.emplace(num);
        isPresent.emplace(num);
    }

private:
    int curr = 1;
    std::priority_queue<int, std::vector<int>, std::greater<int>> added;
    std::unordered_set<int> isPresent;
};

class TreeSet
{
public:
    TreeSet() = default;
    
    int popSmallest()
    {
        int ans;

        if (!added.empty())
        {
            ans = *added.begin();
            added.erase(added.begin());
        }
        else
        {
            ans = curr++;
        }

        return ans;
    }
    
    void addBack(int num)
    {
        if (curr <= num || added.contains(num)) return;
        added.emplace(num);
    }

private:
    int curr = 1;
    std::set<int> added;
};

using SmallestInfiniteSet = HashSetAndHeap;
// using SmallestInfiniteSet = TreeSet;

/**
 * Your SmallestInfiniteSet object will be instantiated and called as such:
 * SmallestInfiniteSet* obj = new SmallestInfiniteSet();
 * int param_1 = obj->popSmallest();
 * obj->addBack(num);
 */