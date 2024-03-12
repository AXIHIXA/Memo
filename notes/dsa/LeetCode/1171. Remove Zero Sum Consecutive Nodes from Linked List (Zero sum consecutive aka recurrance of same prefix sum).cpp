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
    ListNode * removeZeroSumSublists(ListNode * head)
    {
        ListNode pivot(0);
        pivot.next = head;

        int prefixSum = 0;
        std::unordered_map<int, ListNode *> hashMap;

        // curr starts from &pivot s.t. hashMap.at(0) == &pivot. 
        // Fixes testcase [1, -1]. 
        for (ListNode * curr = &pivot; curr; curr = curr->next)
        {
            prefixSum += curr->val;

            auto it = hashMap.find(prefixSum);

            if (it != hashMap.end())
            {
                int p = prefixSum;
                ListNode * pred = it->second;
                
                for (ListNode * q = pred->next; q != curr; q = q->next)
                {
                    p += q->val;
                    hashMap.erase(p);
                }

                pred->next = curr->next;
            }
            else
            {
                hashMap.emplace(prefixSum, curr);
            }
        }

        return pivot.next;
    }
};