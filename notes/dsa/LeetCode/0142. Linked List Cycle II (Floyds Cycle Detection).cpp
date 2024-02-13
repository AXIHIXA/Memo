/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution
{
public:
    ListNode * detectCycle(ListNode * head)
    {
        if (!head) return nullptr;

        ListNode * slow = head;
        ListNode * fast = head;

        do 
        {
            slow = slow->next;
            fast = fast->next;
            if (fast) fast = fast->next;
        }
        while (slow && fast && (slow != fast));

        if (!slow || !fast) return nullptr;

        slow = head;

        while (slow != fast)
        {
            slow = slow->next;
            fast = fast->next;
        }

        return fast;
    }
};