class Solution
{
public:
    int numRescueBoats(std::vector<int> & people, int limit)
    {
        std::sort(people.begin(), people.end());
        int ans = 0;
        
        for (int i = 0, j = people.size() - 1; i <= j; --j, ++ans)
        {
            if (people[i] + people[j] <= limit)
            {
                ++i;
            }
        }

        return ans;
    }
};

/*
使用「归纳法」证明猜想的正确性。

假设最优成船组合中二元组的数量为 c1，我们贪心做法的二元组数量为 c2。

最终答案 = 符合条件的二元组的数量 + 剩余人数数量，而在符合条件的二元组数量固定的情况下，剩余人数也固定。因此我们只需要证明 c1=c2 即可。

通常使用「归纳法」进行证明，都会先从边界入手。

当我们处理最重的人 people[r]（此时 r 为原始右边界 n−1 ）时：

假设其与 people[l]（此时 l 为原始左边界 0）之和超过 limit，说明 people[r] 与数组任一成员组合都会超过 limit，即无论在最优组合还是贪心组合中，people[r] 都会独立成船；

假设 people[r]+people[l]<=limit，说明数组中存在至少一个成员能够与 people[l] 成船：

假设在最优组合中 people[l] 独立成船，此时如果将贪心组合 (people[l],people[r]) 中的 people[l] 拆分出来独立成船，贪心二元组数量 c2 必然不会变大（可能还会变差），即将「贪心解」调整成「最优解」结果不会变好；

假设在最优组合中，people[l] 不是独立成船，又因此当前 r 处于原始右边界，因此与 people[l] 成组的成员 people[x] 必然满足 people[x]<=people[r]。
此时我们将 people[x] 和 people[r] 位置进行交换（将贪心组合调整成最优组合），此时带来的影响包括：

与 people[l] 成组的对象从 people[r] 变为 people[x]，但因为 people[x]<=people[r]，即有 people[l]+people[x]<=people[l]+people[r]<=limit，仍为合法二元组，消耗船的数量为 1；
原本位置 x 的值从 people[x] 变大为 people[r]，如果调整后的值能组成二元组，那么原本更小的值也能组成二元组，结果没有变化；如果调整后不能成为组成二元组，那么结果可能会因此变差。
综上，将 people[x] 和 people[r] 位置进行交换（将贪心组合调整成最优组合），贪心二元组数量 c2 不会变大，即将「贪心解」调整成「最优解」结果不会变好。

对于边界情况，我们证明了从「贪心解」调整为「最优解」不会使得结果更好，因此可以保留当前的贪心决策，然后将问题规模缩减为 n−1 或者 n−2，同时数列仍然满足升序特性，即归纳分析所依赖的结构没有发生改变，可以将上述的推理分析推广到每一个决策的回合（新边界）中。

至此，我们证明了将「贪心解」调整为「最优解」结果不会变好，即贪心解是最优解之一。

作者：宫水三叶
链接：https://leetcode.cn/problems/boats-to-save-people/solutions/1/gong-shui-san-xie-noxiang-xin-ke-xue-xi-hosg8/
*/