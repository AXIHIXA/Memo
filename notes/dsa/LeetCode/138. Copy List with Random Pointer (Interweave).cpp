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
        if (!head)
        {
            return head;
        }
        
        // Interweave new nodes into original list
        for (Node * node = head; node; )
        {
            Node * newNode = new Node(node->val);
            Node * nodeNext = node->next;

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
        Node * node = head;
        Node * newNode = head->next;
        Node * newHead = head->next;

        while (newNode->next)
        {
            node->next = newNode->next;
            node = node->next;

            newNode->next = node->next;
            newNode = newNode->next;
        }

        node->next = nullptr;

        return newHead;
    }
};
