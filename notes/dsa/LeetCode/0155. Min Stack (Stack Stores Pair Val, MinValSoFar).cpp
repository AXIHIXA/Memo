class MinStack 
{
public:
    MinStack() = default;
    
    void push(int val) 
    {
        st.emplace(val, st.empty() ? val : std::min(val, st.top().second));
    }
    
    void pop() 
    {
        st.pop();
    }
    
    int top() 
    {
        return st.top().first;
    }
    
    int getMin() 
    {
        return st.top().second;
    }

private:
    std::stack<std::pair<int, int>> st;
};

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack* obj = new MinStack();
 * obj->push(val);
 * obj->pop();
 * int param_3 = obj->top();
 * int param_4 = obj->getMin();
 */