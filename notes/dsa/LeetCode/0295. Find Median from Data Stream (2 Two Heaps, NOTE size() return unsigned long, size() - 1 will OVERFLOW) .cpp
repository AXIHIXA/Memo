class MedianFinder 
{
public:
    MedianFinder() = default;
    
    void addNum(int num) 
    {
        if (lo.empty() || num <= lo.top()) lo.emplace(num);
        else                               hi.emplace(num);

        // NOTE THAT size() returns unsigned long!!!
        // size() - 1 will OVERFLOW!!!

        if (lo.size() + 1 < hi.size()) 
        {
            lo.emplace(hi.top());
            hi.pop();
        }

        if (hi.size() + 1 < lo.size())
        {
            hi.emplace(lo.top());
            lo.pop();
        }
    }
    
    double findMedian() 
    {
        if (lo.size() < hi.size())      return hi.top();
        else if (hi.size() < lo.size()) return lo.top();
        else return static_cast<double>(lo.top() + hi.top()) * 0.5;
    }

private:
    std::priority_queue<int> lo;
    std::priority_queue<int, std::vector<int>, std::greater<int>> hi;
};

/**
 * Your MedianFinder object will be instantiated and called as such:
 * MedianFinder* obj = new MedianFinder();
 * obj->addNum(num);
 * double param_2 = obj->findMedian();
 */