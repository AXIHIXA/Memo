#include <bits/stdc++.h>


class Solution
{
public:
    int electricCarPlan(
            std::vector<std::vector<int>> & paths,
            int cnt,
            int start,
            int end,
            std::vector<int> & charge)
    {
        auto n = static_cast<int>(charge.size());
        auto m = static_cast<int>(paths.size());

        std::vector g(n, std::vector<std::pair<int, int>>());

        for (const auto & path : paths)
        {
            g[path[0]].emplace_back(path[1], path[2]);
            g[path[1]].emplace_back(path[0], path[2]);
        }

        auto cmp =
        [](const std::tuple<int, int, int> & a,
           const std::tuple<int, int, int> & b) -> bool
        {
            return std::get<2>(a) > std::get<2>(b);
        };

        std::priority_queue<
                std::tuple<int, int, int>,
                std::vector<std::tuple<int, int, int>>,
                decltype(cmp)
        > heap;

        heap.emplace(start, 0, 0);
        // std::printf("Push %d %d %d\n", start, 0, 0);

        // visited, dist[v][c] denotes arriving at node v with c units of power remaining.
        std::vector visited(n, std::vector<std::uint8_t>(cnt + 1, false));
        std::vector dist(n, std::vector<int>(cnt + 1, std::numeric_limits<int>::max()));
        dist[start][0] = 0;

        while (!heap.empty())
        {
            auto [s, sPow, sDist] = heap.top();
            heap.pop();
            // std::printf("Pop  %d %d %d\n", s, sPow, sDist);

            if (visited[s][sPow])
            {
                continue;
            }

            visited[s][sPow] = true;

            if (s == end)
            {
                return sDist;
            }

            // Charge one unit of power.
            // 2+ units of power is done by node added this time.
            // Note that Dijkstra's algorithm expands only the smallest node.
            if (sPow < cnt)
            {
                int t = s;
                int tPow = sPow + 1;
                int tDist = sDist + charge[s];

                // WARNING!!!
                // USE CONTINUE ONLY IMMEDIATELY INSIDE LOOPS!!!
                // A CONTINUE HERE WILL SKIP EDGE EXPANSIONS!!!
                if (!visited[t][tPow] && tDist < dist[t][tPow])
                {
                    dist[t][tPow] = tDist;
                    heap.emplace(t, tPow, tDist);
                    // std::printf("Push %d %d %d\n", t, tPow, tDist);
                }
            }

            // Move to an adjacency node without charging.
            for (auto [t, w] : g[s])
            {
                int tPow = sPow - w;
                int tDist = sDist + w;

                if (tPow < 0 || visited[t][tPow] || dist[t][tPow] <= tDist)
                {
                    continue;
                }

                dist[t][tPow] = tDist;
                heap.emplace(t, tPow, tDist);
                // std::printf("Push %d %d %d\n", t, tPow, tDist);
            }
        }

        return -1;
    }
};
