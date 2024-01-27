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
    ListNode * reverseKGroup(ListNode * head, int k)
    {
        if (!head->next || k == 1) return head;

        ListNode * ptr = head;
        ListNode * tail = nullptr;
        ListNode * newHead = nullptr;

        while (ptr)
        {
            int count = 0;
            ptr = head;

            while (count < k && ptr != nullptr)
            {
                ptr = ptr->next;
                ++count;
            }

            if (count == k)
            {
                ListNode * revHead = reverseList(head, k);
                if (!newHead) newHead = revHead;
                if (tail) tail->next = revHead;

                tail = head;
                head = ptr;
            }
        }

        if (tail) tail->next = head;

        return newHead ? newHead : head;
    }

private:
    static ListNode * reverseList(ListNode * head, int k)
    {
        ListNode * p1 = nullptr;
        ListNode * p2 = head;

        while (0 < k--)
        {
            ListNode * tmp = p2->next;
            p2->next = p1;
            p1 = p2;
            p2 = tmp;
        }

        return p1;
    }
};