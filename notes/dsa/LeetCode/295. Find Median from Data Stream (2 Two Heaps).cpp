class MedianFinder 
{
public:
    MedianFinder() 
    {
        
    }
    
    void addNum(int num) 
    {
        if (lo.empty() or num <= lo.top())
        {
            lo.push(num);
        }
        else
        {
            hi.push(num);
        }

        if (lo.size() + 1 < hi.size())
        {
            lo.push(hi.top());
            hi.pop();
        }
        else if (hi.size() + 1 < lo.size())
        {
            hi.push(lo.top());
            lo.pop();
        }
    }
    
    double findMedian() 
    {
        if (lo.size() == hi.size())
        {
            return static_cast<double>(lo.top() + hi.top()) * 0.5;
        }
        else if (lo.size() < hi.size())
        {
            return hi.top();
        }
        else
        {
            return lo.top();
        }
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