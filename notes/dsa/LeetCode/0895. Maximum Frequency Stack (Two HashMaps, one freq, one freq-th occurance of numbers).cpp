class FreqStack
{
public:
    using K = int;
    using Freq = int;

public:
    FreqStack() = default;
    
    void push(K x)
    {
        Freq freq = ++frequency[x];
        group[freq].emplace(x);
        maxFreq = std::max(maxFreq, freq);
    }
    
    K pop()
    {
        K x = group.at(maxFreq).top();
        group.at(maxFreq).pop();
        --frequency.at(x);
        if (group.at(maxFreq).empty()) --maxFreq;
        return x;
    }

private:
    int maxFreq = 0;

    // frequency.at(x): Frequency of number x. 
    std::unordered_map<K, Freq> frequency;

    // group.at(freq): Stack containing the freq-th occurances of numbers. 
    std::unordered_map<Freq, std::stack<K>> group;
};

/**
 * Your FreqStack object will be instantiated and called as such:
 * FreqStack* obj = new FreqStack();
 * obj->push(val);
 * int param_2 = obj->pop();
 */