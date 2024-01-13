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
    ListNode * addTwoNumbers(ListNode * l1, ListNode * l2)
    {
        ListNode * head = new ListNode(0);
        ListNode * p = head;

        int increment = 0;
        int digit = 0;

        while (true)
        {
            int v1 = l1 ? l1->val : 0;
            int v2 = l2 ? l2->val : 0;
            
            p->val = v1 + v2 + increment;

            if (9 < p->val)
            {
                p->val %= 10;
                increment = 1;
            }
            else
            {
                increment = 0;
            }

            if (l1) l1 = l1->next;
            if (l2) l2 = l2->next;

            if (l1 or l2 or increment)
            {
                p->next = new ListNode(0);
                p = p->next;
            }
            else
            {
                break;
            }
        }

        return head;
    }
};