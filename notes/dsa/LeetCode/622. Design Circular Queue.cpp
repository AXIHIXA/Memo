class MyCircularQueue 
{
public:
    MyCircularQueue(int k) 
    {
        elem = new int[k];
        capacity = k;
        headIndex = 0;
        size = 0;
    }
    
    bool enQueue(int value) 
    {
        if (size == capacity)
        {
            return false;
        }

        ++size;
        elem[(headIndex + size - 1) % capacity] = value;
        return true;
    }
    
    bool deQueue() 
    {
        if (!size)
        {
            return false;
        }

        --size;
        ++headIndex;
        headIndex %= capacity;
        return true;
    }
    
    int Front()
    {
        return size ? elem[headIndex] : -1;
    }
    
    int Rear() 
    {
        return size ? elem[(headIndex + size - 1) % capacity] : -1;
    }
    
    bool isEmpty() 
    {
        return !size;
    }
    
    bool isFull()
    {
        return size == capacity;
    }

private:
    int * elem {nullptr};
    int capacity {0};
    int headIndex {0};
    int size {0};
};

/**
 * Your MyCircularQueue object will be instantiated and called as such:
 * MyCircularQueue* obj = new MyCircularQueue(k);
 * bool param_1 = obj->enQueue(value);
 * bool param_2 = obj->deQueue();
 * int param_3 = obj->Front();
 * int param_4 = obj->Rear();
 * bool param_5 = obj->isEmpty();
 * bool param_6 = obj->isFull();
 */