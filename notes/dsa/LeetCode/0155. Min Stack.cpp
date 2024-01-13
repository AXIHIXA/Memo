class MinStack 
{
public:
    MinStack() = default;
    
    void push(int val) 
    {
        st.emplace(val, st.empty() ? val : min(val, st.top().minValSoFar));
    }
    
    void pop() 
    {
        st.pop();
    }
    
    int top() 
    {
        return st.top().val;
    }
    
    int getMin() 
    {
        return st.top().minValSoFar;
    }

private:
    struct Node 
    {
        Node() = default;
        Node(int v, int m) : val(v), minValSoFar(m) {}

        int val {0};
        int minValSoFar {0};
    };

    stack<Node> st;
};

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack* obj = new MinStack();
 * obj->push(val);
 * obj->pop();
 * int param_3 = obj->top();
 * int param_4 = obj->getMin();
 */