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
    ListNode * mergeTwoLists(ListNode * list1, ListNode * list2)
    {
        std::unique_ptr<ListNode> pp = std::make_unique<ListNode>();
        ListNode * i = pp.get(), * j = list1, * k = list2;

        while (j && k)
        {
            if (j && j->val <= k->val)
            {
                i->next = j;
                j = j->next;
            }
            else
            {
                i->next = k;
                k = k->next;
            }

            i = i->next;
        }

        i->next = j ? j : k;

        return pp->next;
    }
};