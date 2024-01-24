/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution
{
public:
    ListNode * reverseBetween(ListNode * head, int left, int right)
    {
        if (left == right) return head;
        if (!head) return nullptr;
        
        // Make pred points to left - 1 and curr points to left. 
        ListNode * curr = head;
        ListNode * pred = nullptr;

        while (1 < left)
        {
            pred = curr;
            curr = curr->next;
            --left, --right;
        }

        // Save pred and curr to fix the final connections.
        ListNode * ll = pred;
        ListNode * rr = curr;

        // Iteratively reverse the nodes.
        for (ListNode * tmp = nullptr; 0 < right; --right)
        {
            tmp = curr->next;
            curr->next = pred;
            pred = curr;
            curr = tmp;
        }

        // Adjust the final connections. 
        if (ll) ll->next = pred;
        else    head = pred;

        rr->next = curr;

        return head;
    }
};