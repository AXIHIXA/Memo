#include <fstream>
#include <vector>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Surface_mesh/IO.h>

#include <fmt/core.h>


int main(int argc, char * argv[])
{
    std::string testcase = argv[1];
    int size = 1025;
    int size2 = size * size;

    std::vector<int> xv, yv, mv;
    std::vector<float> hv;

    xv.reserve(size2);
    yv.reserve(size2);
    mv.reserve(size2);
    hv.reserve(size2);

    typedef CGAL::Simple_cartesian<float> K;
    typedef K::Point_3 Point;
    typedef CGAL::Surface_mesh<Point> Mesh;
    typedef Mesh::Vertex_index Vertex;
    typedef Mesh::Face_index Face;

    Mesh mesh;

    std::vector<Vertex> vd(size2, Vertex());

    if (std::ifstream fin {fmt::format("var/txt/{}_{}.txt", testcase, size)})
    {
        int x, y, m;
        float z;

        while (fin >> x >> y >> m >> z)
        {
            xv.emplace_back(x);
            yv.emplace_back(y);
            mv.emplace_back(m);
            hv.emplace_back(z);

            vd.at(x * size + y) = mesh.add_vertex({y, -x, z * std::stof(argv[2])});
        }
    }
    else
    {
        throw std::runtime_error(fmt::format("failed to open file {}:{}", __FILE__, __LINE__));
    }

    for (int i = 0, k = 0; i != size; ++i)
    {
        for (int j = 0; j != size; ++j, ++k)
        {
            // Add a bottom-right square (two triangular faces) iff.:
            //     1. This square exists on the grid;
            //     2. All four vertices are interior nodes.

            if (i != size - 1 and j != size - 1)
            {
                if (mv[k] == 0 and mv[k + 1] == 0 and mv[k + size] == 0 and mv[k + size + 1] == 0)
                {
                    mesh.add_face(vd[k], vd[k + size], vd[k + 1]);
                    mesh.add_face(vd[k + size], vd[k + size + 1], vd[k + 1]);
                }
            }
        }
    }

    CGAL::IO::write_PLY(fmt::format("var/ply/{}_{}.ply", testcase, size), mesh);

    return EXIT_SUCCESS;
}

