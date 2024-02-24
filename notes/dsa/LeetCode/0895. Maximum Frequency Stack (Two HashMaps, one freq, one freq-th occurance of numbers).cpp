class FreqStack
{
public:
    FreqStack() = default;
    
    void push(int x)
    {
        maxFreq = std::max(maxFreq, ++freq[x]);
        group[freq.at(x)].emplace(x);
    }
    
    int pop()
    {
        std::stack<int> & g = group.at(maxFreq);
        int x = g.top();
        g.pop();
        if (group.at(freq.at(x)--).empty()) --maxFreq;
        return x;
    }

private:
    int maxFreq = 0;

    // freq.at(x): Frequency of number x. 
    std::unordered_map<int, int> freq;

    // group.at(f): Stack containing the f-th occurances of numbers. 
    std::unordered_map<int, std::stack<int>> group;
};

/**
 * Your FreqStack object will be instantiated and called as such:
 * FreqStack* obj = new FreqStack();
 * obj->push(val);
 * int param_2 = obj->pop();
 */