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
class Solution {
public:
    ListNode* partition(ListNode* head, int x) {
        ListNode * h1 = new ListNode(), * c1 = h1;
        ListNode * h2 = new ListNode(), * c2 = h2;

        while (head)
        {
            if (head->val < x)
            {
                c1->next = head;
                c1 = c1->next;
            }
            else
            {
                c2->next = head;
                c2 = c2->next;
            }

            head = head->next;
        }

        if (!h2->next)
        {
            c1->next = nullptr;
            return h1->next;
        }

        if (!h1->next)
        {
            c2->next = nullptr;
            return h2->next;
        }

        c1->next = h2->next;
        c2->next = nullptr;
        ListNode * ans = h1->next;
        delete h1;
        delete h2;
        return ans;
    }
};