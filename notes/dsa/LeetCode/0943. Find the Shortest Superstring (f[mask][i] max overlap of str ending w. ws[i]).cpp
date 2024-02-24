class Solution
{
public:
    std::string shortestSuperstring(std::vector<std::string> & ws)
    {
        auto n = static_cast<const int>(ws.size());

        // g[i][j]: Length of overlap between ws[i]'s suffix and ws[j]'s prefix. 
        std::vector g(n, std::vector<int>(n, 0));

        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                if (i == j)
                {
                    g[i][j] = ws[i].size();
                    continue;
                }

                for (std::size_t k = std::min(ws[i].size(), ws[j].size()); 1UL <= k; --k)
                {
                    std::string_view si(ws[i].end().base() - k, k);
                    std::string_view pj(ws[j].begin().base(), k);

                    if (si == pj)
                    {
                        g[i][j] = k;
                        break;
                    }
                }
            }
        }

        // mask: 
        //     Status mask: k-th bit == 1 means ws[k] is already included in the superstr. 
        //     mask is the upper limit of status mask (not inclusive). 
        // f[s][i]: 
        //     Max overlap length of superstr: 
        //     (1) with status s; 
        //     (2) ending with ws[i]. 
        // p[s][j] == i:
        //     Aux data to record the string choices for the optimal f[s][j]. 
        //     It means that current optimal f[s][j] comes from f[s'][i], i.e., 
        //     its last two substrs are ws[i] and ws[j], 
        //     (where: (1) s & (1 << j) == 0; AND (2) s' == s | (1 << j)).    
        const int mask = (1 << n);
        std::vector f(mask, std::vector<int>(n, 0));
        std::vector p(mask, std::vector<int>(n, 0));

        for (int s = 1; s < mask; ++s)
        {
            // Loop for all combination of used words. 
            // Try concat ...ws[i] with ws[j]. 
            
            for (int i = 0; i < n; ++i)
            {
                // ws[i] must be already in the current string. 
                if (!(s & (1 << i))) continue;

                for (int j = 0; j < n; ++j)
                {
                    // ws[j] must not be in the string. 
                    if (s & (1 << j)) continue;

                    // NOTE: "<=" REQUIRED, 
                    // because we need to update p info at least once!
                    if (f[s | (1 << j)][j] <= f[s][i] + g[i][j])
                    {
                        f[s | (1 << j)][j] = f[s][i] + g[i][j];
                        p[s | (1 << j)][j] = i;
                    }
                }
            }
        }

        // Try to assemble the superstr from the last ws to the first. 
        // The last ws[i] is the argmax_i f[mask - 1][i]. 
        int idx = 0;
        int maxOverlap = f[mask - 1][0];

        for (int i = 1; i < n; ++i)
        {
            if (maxOverlap < f[mask - 1][i])
            {
                maxOverlap = f[mask - 1][i];
                idx = i;
            }
        }

        std::string ans;

        // NOTE: 
        //     s MUST start with mask - 1 and go down to 0, 
        //     so that we could query p[s][idx]!
        for (int s = mask - 1, last = -1; s; )
        {
            if (last == -1) ans = ws[idx];
            else ans = ws[idx].substr(0, ws[idx].size() - g[idx][last]) + ans;

            last = idx;
            idx = p[s][idx];
            s &= ~(1 << last);
        }

        return ans;
    }
};