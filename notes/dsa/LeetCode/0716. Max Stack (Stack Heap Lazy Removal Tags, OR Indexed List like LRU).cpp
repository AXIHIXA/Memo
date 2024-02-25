class HeapStackLazyRemoval
{
public:
    HeapStackLazyRemoval() = default;
    
    void push(int x)
    {
        heap.emplace(x, timeStamp);
        st.emplace(x, timeStamp);
        ++timeStamp;
    }
    
    int pop()
    {
        while (removed.contains(st.top().second)) st.pop();
        auto [x, i] = st.top();
        st.pop();
        removed.emplace(i);
        return x;
    }
    
    int top()
    {
        while (removed.contains(st.top().second)) st.pop();
        return st.top().first;
    }
    
    int peekMax()
    {
        while (removed.contains(heap.top().second)) heap.pop();
        return heap.top().first;
    }
    
    int popMax()
    {
        while (removed.contains(heap.top().second)) heap.pop();
        auto [x, i] = heap.top();
        heap.pop();
        removed.emplace(i);
        return x;
    }

private:
    int timeStamp = 0;

    // {value, id of this value (size when pushing)}
    std::priority_queue<std::pair<int, int>> heap;
    std::stack<std::pair<int, int>> st;

    // Lazy removal: IDs of removed values. 
    std::unordered_set<int> removed;
};

class IndexedList
{
public:
    IndexedList() = default;
    
    void push(int x)
    {
        lst.emplace_front(x, timeStamp++);
        tree.emplace(lst.front(), lst.begin());
    }
    
    int pop()
    {
        auto top = lst.front();
        tree.erase(top);
        lst.pop_front();
        return top.first;
    }
    
    int top()
    {
        return lst.front().first;
    }
    
    int peekMax()
    {
        return tree.rbegin()->first.first;
    }
    
    int popMax()
    {
        auto it = tree.rbegin()->second;
        int x = it->first;
        lst.erase(it);

        // NOTE: 
        // For back element access, 
        //     we could do both *(tree.rbegin()) and *(std::prev(tree.end())); 
        // but if we need iterator to back element,
        //     we CAN'T go tree.rbegin().base() as reversed iterators are shifted, 
        //     it will shift back to tree.end() and cause segmentation faults!!!
        tree.erase(std::prev(tree.end()));

        return x;
    }

private:
    int timeStamp = 0;
    std::list<std::pair<int, int>> lst;
    std::map<std::pair<int, int>, std::list<std::pair<int, int>>::iterator> tree;
};

// using MaxStack = HeapStackLazyRemoval;
using MaxStack = IndexedList;

/**
 * Your MaxStack object will be instantiated and called as such:
 * MaxStack* obj = new MaxStack();
 * obj->push(x);
 * int param_2 = obj->pop();
 * int param_3 = obj->top();
 * int param_4 = obj->peekMax();
 * int param_5 = obj->popMax();
 */