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
        ListNode * i = pp.get();
        ListNode * j = list1;
        ListNode * k = list2;

        while (j && k)
        {
            if (j && (!k || j->val <= k->val))
            {
                i->next = j;
                i = i->next;
                j = j->next;
            }

            if (k && (!j || k->val < j->val))
            {
                i->next = k;
                i = i->next;
                k = k->next; 
            }
        }

        if (j || k) i->next = (j ? j : k);

        return pp->next;
    }
};