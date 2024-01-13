class MyLinkedList 
{
public:
    MyLinkedList() : head(new Node), tail(new Node) 
    {
        head->next = tail;
        tail->prev = head;
    }
    
    int get(int index) 
    {
        Node * p = getNode(index);
        return p ? p->val : -1;
    }
    
    void addAtHead(int val) 
    {
        insertBetween(val, head, head->next);
    }
    
    void addAtTail(int val) 
    {
        insertBetween(val, tail->prev, tail);
    }
    
    void addAtIndex(int index, int val) 
    {
        Node * succ = index == sz ? tail : getNode(index);

        if (succ)
        {
            insertBetween(val, succ->prev, succ);
        }
    }
    
    void deleteAtIndex(int index) 
    {
        Node * p = getNode(index);

        if (p)
        {
            Node * pred = p->prev;
            Node * succ = p->next;
            delete p;
            --sz;
            pred->next = succ;
            succ->prev = pred;
        }
    }

private:
    struct Node
    {
        int val {0};
        Node * next {nullptr};
        Node * prev {nullptr};
    };

    void insertBetween(int val, Node * pred, Node * succ)
    {
        Node * p = new Node;
        p->val = val;

        pred->next = p;
        p->prev = pred;

        succ->prev = p;
        p->next = succ;

        ++sz;
    }

    Node * getNode(int index)
    {
        if (not (0 <= index and index < sz))
        {
            return nullptr;
        }

        Node * p = head->next;

        for (int i = 0; i != index; ++i)
        {
            p = p->next;
        }

        return p;
    }

    void print()
    {
        for (Node * p = head->next; p != tail; p = p->next)
        {
            cout << p->val << ' ';
        }

        cout << "\n=======\n";
    }

    int sz = 0;
    Node * head {nullptr};
    Node * tail {nullptr};
};

/**
 * Your MyLinkedList object will be instantiated and called as such:
 * MyLinkedList* obj = new MyLinkedList();
 * int param_1 = obj->get(index);
 * obj->addAtHead(val);
 * obj->addAtTail(val);
 * obj->addAtIndex(index,val);
 * obj->deleteAtIndex(index);
 */