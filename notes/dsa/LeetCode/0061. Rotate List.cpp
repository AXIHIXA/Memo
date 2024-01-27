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
    ListNode * rotateRight(ListNode * head, int k)
    {
        if (k == 0 || !head || !head->next) return head;

        ListNode * ptr = head;
        int len = 0;

        while (ptr)
        {
            ptr = ptr->next;
            ++len;
        }

        k %= len;
        if (k == 0) return head;

        // fast == slow + k
        ListNode * slow = head;
        ListNode * fast = head;
        while (k--) fast = fast->next;
        
        // fast points to the last node (NOT nullptr)
        while (fast->next)
        {
            slow = slow->next;
            fast = fast->next;
        }

        ListNode * newHead = slow->next;
        slow->next = nullptr;

        ptr = newHead;
        while (ptr->next) ptr = ptr->next;
        ptr->next = head;

        return newHead;
    }
};