class MedianFinder 
{
public:
    MedianFinder() 
    {
        
    }
    
    // A BST. 
    // Suppose midIt points to median or next(median) for odd-numbered tree. 
    // A new node gets inserted, midIt only needs to be moved by 1 step! 
    void addNum(int num) 
    {
        auto n = static_cast<int>(tree.size());
        tree.insert(num);

        if (!n)
        {
            midIt = tree.begin();
        }
        else
        {
            if (num < *midIt)
            {
                if (!(n & 1))
                {
                    midIt = prev(midIt);
                }
            }
            else
            {
                if (n & 1)
                {
                    midIt = next(midIt);
                }
            }
        }

        cout << "midIt -> " << *midIt << '\n';
    }
    
    double findMedian() 
    {
        if (tree.size() & 1)
        {
            return *midIt;
        }
        
        double b = *midIt;
        double a = *(prev(midIt));

        return (a + b) * 0.5;
    }

private:
    std::multiset<int> tree;
    std::multiset<int>::iterator midIt;
};

/**
 * Your MedianFinder object will be instantiated and called as such:
 * MedianFinder* obj = new MedianFinder();
 * obj->addNum(num);
 * double param_2 = obj->findMedian();
 */