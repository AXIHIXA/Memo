class Solution
{
public:
    std::string simplifyPath(std::string path)
    {
        std::vector<std::string> blocks;
        std::stringstream sin(path);

        for (std::string line; std::getline(sin, line, '/'); )
        {
            if (line.empty() || line == ".") continue;
            else if (line == "..") { if (!blocks.empty()) blocks.pop_back(); }
            else blocks.push_back(line);
        }

        std::string ans = "/";
        for (int i = 0; i < blocks.size(); ++i)
            ans += blocks[i] + "/";

        return ans.size() == 1 ? "/" : ans.substr(0, ans.size() - 1);
    }
};