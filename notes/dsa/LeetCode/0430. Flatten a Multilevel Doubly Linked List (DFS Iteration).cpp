/*
// Definition for a Node.
class Node {
public:
    int val;
    Node* prev;
    Node* next;
    Node* child;
};
*/

class Solution
{
public:
    Node * flatten(Node * head)
    {
        if (!head) return nullptr;

        std::stack<Node *> st;
        st.emplace(head);

        std::unique_ptr<Node> buf = std::make_unique<Node>();
        Node * ans = buf.get();
        Node * prev = buf.get();

        while (!st.empty())
        {
            Node * curr = st.top();
            st.pop();

            prev->next = curr;
            curr->prev = prev;
            prev = curr;

            if (curr->next) st.emplace(curr->next);

            if (curr->child)
            {
                st.emplace(curr->child);
                curr->child = nullptr;
            }
        }
        
        ans->next->prev = nullptr;
        return ans->next;
    }
};