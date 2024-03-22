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
    ListNode * mergeInBetween(ListNode * list1, int a, int b, ListNode * list2)
    {
        ListNode * p = list1;
        for (int i = 1; i < a; ++i) p = p->next;
        // p->next is a-th node. 

        ListNode * q = p->next;
        for (int i = a; i <= b; ++i) q = q->next;
        // q is b-th node's next. 
        
        p->next = list2;

        while (list2->next) list2 = list2->next;
        list2->next = q;

        return list1;
    }
};