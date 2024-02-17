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
    bool isPalindrome(ListNode * head)
    {
        if (!head->next) return true;

        int len = 0;
        for (ListNode * p = head; p; p = p->next, ++len); 
        
        ListNode pivot(0);
        pivot.next = head;

        ListNode * slow = &pivot;
        ListNode * fast = &pivot;

        while (fast && fast->next)
        {
            slow = slow->next;
            fast = fast->next->next;
        }

        ListNode * le;
        ListNode * rb;

        if (len & 1)
        {
            // [ ... ] slow [ ... ]
            le = slow;
            rb = slow->next;
        }
        else
        {
            // [ ... slow] [slow->next ... ]
            le = slow->next;
            rb = slow->next;
        }

        rb = reverse(rb);

        for (; head != le && rb; head = head->next, rb = rb->next)
        {
            if (head->val != rb->val) return false;
        }

        return head == le && !rb;
    }

private:
    ListNode * reverse(ListNode * head)
    {
        ListNode * prev = nullptr;
        ListNode * next = nullptr;

        while (head)
        {
            next = head->next;
            head->next = prev;
            prev = head;

            if (!next) break;
            head = next;
        }

        return head;
    }
};