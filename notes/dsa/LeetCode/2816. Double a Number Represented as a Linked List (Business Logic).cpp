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
    ListNode * doubleIt(ListNode * head) 
    {
        if (head->val == 0)
        {
            return head;
        }

        head = reverse(head);

        int increment = 0;
        ListNode * p = head;

        while (true)
        {
            int val = p->val;
            int newVal = (val << 1) + increment;

            if (newVal < 10)
            {
                increment = 0;
                p->val = newVal;
            }
            else  // maxNewVal == 18 + 1
            {
                p->val = newVal % 10;
                increment = 1;
            }

            if (p->next)
            {
                p = p->next;
            }
            else
            {
                break;
            }
        }

        if (increment)
        {
            ListNode * tail = new ListNode(1, nullptr);
            p->next = tail;
        }

        return reverse(head);
    }

    ListNode * reverse(ListNode * head)
    {
        if (!head || !head->next)
        {
            return head;
        }
        
        ListNode * p = head;
        ListNode * pred = nullptr; 
        
        while (true)
        {
            ListNode * pNext = p->next;
            p->next = pred;

            if (pNext)
            {
                pred = p;
                p = pNext;
            }
            else  // p is end of list
            {
                return p;
            }
        }
    }
};