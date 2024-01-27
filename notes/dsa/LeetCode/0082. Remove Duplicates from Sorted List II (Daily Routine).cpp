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
    ListNode * deleteDuplicates(ListNode * head)
    {
        if (!head || !head->next) return head;
        
        std::unique_ptr<ListNode> sentinel = std::make_unique<ListNode>(0, head);
        ListNode * pred = sentinel.get();

        while (head)
        {
            if (head->next && head->val == head->next->val)
            {
                while (head->next && head->val == head->next->val) head = head->next;
                head = head->next;
                pred->next = head;
            }
            else
            {
                pred = pred->next;
                head = head->next;
            }
        }

        return sentinel->next;
    }
};
