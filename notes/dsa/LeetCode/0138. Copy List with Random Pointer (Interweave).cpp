/*
// Definition for a Node.
class Node {
public:
    int val;
    Node* next;
    Node* random;
    
    Node(int _val) {
        val = _val;
        next = NULL;
        random = NULL;
    }
};
*/

class Solution 
{
public:
    Node * copyRandomList(Node * head) 
    {
        if (!head) return nullptr;
        
        // Interweave new nodes into original list
        for (Node * node = head; node; )
        {
            Node * nodeNext = node->next;
            Node * newNode = new Node(node->val);
            
            node->next = newNode;
            newNode->next = nodeNext;
            node = nodeNext;
        }

        // Update random pointers of new nodes
        for (Node * node = head; node; node = node->next->next)
        {
            node->next->random = node->random ? node->random->next : nullptr;
        }

        // Unweave the interweaved list
        Node * n1 = head;
        Node * n2 = head->next;
        Node * newHead = head->next;

        while (n2->next)
        {
            n1->next = n2->next;
            n1 = n1->next;

            n2->next = n1->next;
            n2 = n2->next;
        }

        // Nexts for the last nodes in two lists are left unmodified. 
        n1->next = nullptr;

        return newHead;
    }
};
