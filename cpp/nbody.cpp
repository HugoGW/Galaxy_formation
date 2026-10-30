// Barnes-Hut gravity and particle rendering engine, called from Python via ctypes
// (see galaxy_formation/engine.py).
//
// Gravity (G = 1, Plummer softening eps):
// - Octree built by partitioning particle indices. The top of the tree is split into
//   sub-trees that are built in parallel, then stitched together.
// - Nodes are stored in depth-first order with a "next" pointer, so the tree is walked
//   without a stack.
// - Group walk (Barnes 1990): particles that are close in space (a sub-tree with at most
//   GROUP_SIZE particles) share one tree walk. The walk produces an interaction list that
//   is then evaluated for every particle of the group with a vectorised (SIMD) loop.
// - Particles are copied in tree order for memory locality.
// - "Refit" mode: between two rebuilds, the tree keeps its structure and only its masses,
//   centres of mass and opening radii are updated (used for the sub-steps of the
//   hierarchical time integration).
// - Only "active" particles get a new acceleration (hierarchical time steps).
//
// Rendering:
// - nbody_render: orthographic projection, bilinear deposit into one image per colour group.
// - nbody_tonemap: logarithmic tone mapping + glow + soft saturation to 8-bit RGB.
//
// Build (done automatically by engine.py, or with `make`):
//   g++ -O3 -march=native -ffast-math -fopenmp -std=c++17 -shared -fPIC nbody.cpp -o libnbody.so

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <omp.h>

namespace {

constexpr int LEAF_SIZE = 8;    // max particles per leaf
constexpr int GROUP_SIZE = 64;  // max particles per group (one tree walk per group)
constexpr int MAX_DEPTH = 48;
constexpr int TOP_DEPTH = 2;    // depth below which sub-trees are built in parallel

struct Node {
    float cx, cy, cz, m;     // centre of mass and mass
    float open_r;            // opening radius: node accepted if distance > open_r (+ group radius)
    int32_t start, count;    // particles (in tree order) contained in the node
    int32_t next;            // next node outside this sub-tree
    int32_t leaf;            // 1 for a leaf
    float gx, gy, gz, half;  // geometric box (needed by refit)
};

struct Box {
    int b, e;
    double c[3], half;
    int depth;
};

// Top-of-tree node: either internal (children) or a pointer to a parallel sub-tree task.
struct TopNode {
    Box box;
    int task = -1;
    std::vector<int> children;
};

struct Tree {
    std::vector<Node> nodes;
    std::vector<int32_t> idx, tmp, groups;
    std::vector<uint8_t> oct;
    std::vector<float> px, py, pz, pm;   // particles in tree order (single precision)
    std::vector<double> x_build;         // positions when the tree was built
    std::vector<float> disp, node_disp;  // displacement since the build (refit)
    const double* pos = nullptr;
    const double* mass = nullptr;
    float theta = 0.7f;

    std::vector<TopNode> top;
    std::vector<Box> task_box;
    std::vector<std::vector<Node>> task_nodes;

    // Sorts idx[b, e) into the 8 octants around c; off[0..8] are the octant bounds.
    void partition(int b, int e, const double c[3], int off[9]) {
        int cnt[8] = {0};
        for (int i = b; i < e; ++i) {
            int j = idx[i];
            uint8_t o = (pos[3 * j] > c[0]) | ((pos[3 * j + 1] > c[1]) << 1) | ((pos[3 * j + 2] > c[2]) << 2);
            oct[i] = o;
            ++cnt[o];
        }
        off[0] = b;
        for (int o = 0; o < 8; ++o) off[o + 1] = off[o] + cnt[o];
        int fill[8];
        for (int o = 0; o < 8; ++o) fill[o] = off[o];
        for (int i = b; i < e; ++i) tmp[fill[oct[i]]++] = idx[i];
        std::copy(tmp.begin() + b, tmp.begin() + e, idx.begin() + b);
    }

    static void child_center(const double c[3], double h, int o, double cc[3]) {
        cc[0] = c[0] + ((o & 1) ? h : -h);
        cc[1] = c[1] + ((o & 2) ? h : -h);
        cc[2] = c[2] + ((o & 4) ? h : -h);
    }

    void finalize(Node& nd, double M, double sx, double sy, double sz, const Box& bx, bool leaf, int next) {
        double inv = M > 0 ? 1.0 / M : 0.0;
        nd.cx = float(sx * inv);
        nd.cy = float(sy * inv);
        nd.cz = float(sz * inv);
        nd.m = float(M);
        // opening criterion s / (d - delta) < theta, with delta = |centre of mass - box centre|
        double dx = sx * inv - bx.c[0], dy = sy * inv - bx.c[1], dz = sz * inv - bx.c[2];
        nd.open_r = float(2 * bx.half / theta + std::sqrt(dx * dx + dy * dy + dz * dz));
        nd.gx = float(bx.c[0]);
        nd.gy = float(bx.c[1]);
        nd.gz = float(bx.c[2]);
        nd.half = float(bx.half);
        nd.start = bx.b;
        nd.count = bx.e - bx.b;
        nd.leaf = leaf;
        nd.next = next;
    }

    // Recursively builds the sub-tree of bx into out (indices local to out).
    void build_rec(std::vector<Node>& out, const Box& bx) {
        int id = int(out.size());
        out.push_back(Node{});
        double M = 0, sx = 0, sy = 0, sz = 0;
        bool leaf = bx.e - bx.b <= LEAF_SIZE || bx.depth >= MAX_DEPTH;
        if (leaf) {
            for (int i = bx.b; i < bx.e; ++i) {
                int j = idx[i];
                double m = mass[j];
                M += m;
                sx += m * pos[3 * j];
                sy += m * pos[3 * j + 1];
                sz += m * pos[3 * j + 2];
            }
        } else {
            int off[9];
            partition(bx.b, bx.e, bx.c, off);
            double h = 0.5 * bx.half;
            for (int o = 0; o < 8; ++o) {
                if (off[o] == off[o + 1]) continue;
                Box cb{off[o], off[o + 1], {}, h, bx.depth + 1};
                child_center(bx.c, h, o, cb.c);
                int ch = int(out.size());
                build_rec(out, cb);
                const Node& cn = out[ch];
                M += cn.m;
                sx += double(cn.m) * cn.cx;
                sy += double(cn.m) * cn.cy;
                sz += double(cn.m) * cn.cz;
            }
        }
        finalize(out[id], M, sx, sy, sz, bx, leaf, int(out.size()));
    }

    // Step 1 (serial): split the top of the tree into independent tasks.
    int plan(const Box& bx) {
        int id = int(top.size());
        top.emplace_back();
        top[id].box = bx;
        if (bx.depth >= TOP_DEPTH || bx.e - bx.b <= 8 * LEAF_SIZE) {
            top[id].task = int(task_box.size());
            task_box.push_back(bx);
            return id;
        }
        int off[9];
        partition(bx.b, bx.e, bx.c, off);
        double h = 0.5 * bx.half;
        for (int o = 0; o < 8; ++o) {
            if (off[o] == off[o + 1]) continue;
            Box cb{off[o], off[o + 1], {}, h, bx.depth + 1};
            child_center(bx.c, h, o, cb.c);
            int ch = plan(cb);
            top[id].children.push_back(ch);
        }
        return id;
    }

    // Step 3: stitch the sub-trees into `nodes`, in depth-first order.
    int emit(int t) {
        const TopNode& tn = top[t];
        if (tn.task >= 0) {
            const std::vector<Node>& sub = task_nodes[tn.task];
            int offset = int(nodes.size());
            for (Node nd : sub) {
                nd.next += offset;
                nodes.push_back(nd);
            }
            return offset;
        }
        int id = int(nodes.size());
        nodes.push_back(Node{});
        double M = 0, sx = 0, sy = 0, sz = 0;
        for (int c : tn.children) {
            int ch = emit(c);
            const Node& cn = nodes[ch];
            M += cn.m;
            sx += double(cn.m) * cn.cx;
            sy += double(cn.m) * cn.cy;
            sz += double(cn.m) * cn.cz;
        }
        finalize(nodes[id], M, sx, sy, sz, tn.box, false, int(nodes.size()));
        return id;
    }

    void build(int n, const double* pos_, const double* mass_, float theta_) {
        pos = pos_;
        mass = mass_;
        theta = theta_;
        idx.resize(n);
        tmp.resize(n);
        oct.resize(n);

        double x0 = 1e300, y0 = 1e300, z0 = 1e300, x1 = -1e300, y1 = -1e300, z1 = -1e300;
#pragma omp parallel for reduction(min : x0, y0, z0) reduction(max : x1, y1, z1)
        for (int i = 0; i < n; ++i) {
            idx[i] = i;
            x0 = std::min(x0, pos[3 * i]); x1 = std::max(x1, pos[3 * i]);
            y0 = std::min(y0, pos[3 * i + 1]); y1 = std::max(y1, pos[3 * i + 1]);
            z0 = std::min(z0, pos[3 * i + 2]); z1 = std::max(z1, pos[3 * i + 2]);
        }
        Box root{0, n, {0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1)},
                 0.5 * std::max({x1 - x0, y1 - y0, z1 - z0}) * 1.0001 + 1e-12, 0};

        top.clear();
        task_box.clear();
        plan(root);                                   // step 1
        task_nodes.resize(task_box.size());
#pragma omp parallel for schedule(dynamic, 1)
        for (int k = 0; k < int(task_box.size()); ++k) {  // step 2: sub-trees in parallel
            task_nodes[k].clear();
            build_rec(task_nodes[k], task_box[k]);
        }
        nodes.clear();
        nodes.reserve(2 * n / LEAF_SIZE + 1024);
        emit(0);                                      // step 3

        // groups: largest nodes holding at most GROUP_SIZE particles
        groups.clear();
        for (int i = 0; i < int(nodes.size());) {
            if (nodes[i].count <= GROUP_SIZE || nodes[i].leaf) {
                groups.push_back(i);
                i = nodes[i].next;
            } else {
                ++i;
            }
        }

        // copy the particles in tree order
        px.resize(n); py.resize(n); pz.resize(n); pm.resize(n);
        x_build.assign(pos, pos + 3 * n);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            int j = idx[i];
            px[i] = float(pos[3 * j]);
            py[i] = float(pos[3 * j + 1]);
            pz[i] = float(pos[3 * j + 2]);
            pm[i] = float(mass[j]);
        }
    }
};

// Updates the tree without rebuilding it (particles moved a little): same cells, new
// masses and centres of mass, opening radius enlarged by the largest displacement since
// the build so that the opening criterion stays at least as strict.
void refit(Tree& T, int n, const double* pos, const double* mass) {
    T.pos = pos;
    T.mass = mass;
    T.disp.resize(n);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        int j = T.idx[i];
        T.px[i] = float(pos[3 * j]);
        T.py[i] = float(pos[3 * j + 1]);
        T.pz[i] = float(pos[3 * j + 2]);
        T.pm[i] = float(mass[j]);
        double dx = pos[3 * j] - T.x_build[3 * j], dy = pos[3 * j + 1] - T.x_build[3 * j + 1],
               dz = pos[3 * j + 2] - T.x_build[3 * j + 2];
        T.disp[i] = mass[j] > 0 ? float(std::sqrt(dx * dx + dy * dy + dz * dz)) : 0.0f;
    }
    // bottom-up: mass, centre of mass and largest displacement of every node
    const int nn = int(T.nodes.size());
    T.node_disp.resize(nn);
    for (int k = nn - 1; k >= 0; --k) {
        Node& q = T.nodes[k];
        double M = 0, sx = 0, sy = 0, sz = 0;
        float dm = 0;
        if (q.leaf) {
            for (int i = q.start; i < q.start + q.count; ++i) {
                double m = T.pm[i];
                M += m; sx += m * T.px[i]; sy += m * T.py[i]; sz += m * T.pz[i];
                dm = std::max(dm, T.disp[i]);
            }
        } else {
            for (int c = k + 1; c < q.next; c = T.nodes[c].next) {
                const Node& cn = T.nodes[c];
                M += cn.m; sx += double(cn.m) * cn.cx; sy += double(cn.m) * cn.cy; sz += double(cn.m) * cn.cz;
                dm = std::max(dm, T.node_disp[c]);
            }
        }
        T.node_disp[k] = dm;
        double inv = M > 0 ? 1.0 / M : 0.0;
        q.cx = float(sx * inv); q.cy = float(sy * inv); q.cz = float(sz * inv);
        q.m = float(M);
        double dx = sx * inv - q.gx, dy = sy * inv - q.gy, dz = sz * inv - q.gz;
        // the node's particles are at most half + dm (per axis) from the box centre
        q.open_r = float(2 * (q.half + dm) / T.theta + std::sqrt(dx * dx + dy * dy + dz * dz));
    }
}

Tree g_tree;

// Per-thread interaction list: accepted nodes and particles of opened leaves.
struct List {
    std::vector<float> x, y, z, m;
    int n = 0;
    void reset(size_t cap) {
        if (x.size() < cap) { x.resize(cap); y.resize(cap); z.resize(cap); m.resize(cap); }
        n = 0;
    }
    inline void push(float a, float b, float c, float d) {
        if (n == int(x.size())) { size_t s = 2 * x.size(); x.resize(s); y.resize(s); z.resize(s); m.resize(s); }
        x[n] = a; y[n] = b; z[n] = c; m[n] = d; ++n;
    }
};

void compute_accel(int n, const double* pos, const double* mass, double* acc,
                   double theta, double eps, const uint8_t* active, bool rebuild) {
    Tree& T = g_tree;
    if (rebuild || int(T.idx.size()) != n)
        T.build(n, pos, mass, float(theta));
    else
        refit(T, n, pos, mass);
    const float eps2 = float(eps * eps);
    const Node* nd = T.nodes.data();
    const int nn = int(T.nodes.size());
    const int ng = int(T.groups.size());

#pragma omp parallel
    {
        List L;
        L.reset(8192);
#pragma omp for schedule(dynamic, 4)
        for (int g = 0; g < ng; ++g) {
            const Node& grp = nd[T.groups[g]];
            const int b = grp.start, e = grp.start + grp.count;
            // hierarchical time steps: skip groups without any active particle
            if (active) {
                bool any = false;
                for (int i = b; i < e && !any; ++i) any = active[T.idx[i]] != 0;
                if (!any) continue;
            }

            // sphere enclosing the particles of the group
            float x0 = 1e30f, y0 = 1e30f, z0 = 1e30f, x1 = -1e30f, y1 = -1e30f, z1 = -1e30f;
            for (int i = b; i < e; ++i) {
                if (T.pm[i] <= 0) continue;
                x0 = std::min(x0, T.px[i]); x1 = std::max(x1, T.px[i]);
                y0 = std::min(y0, T.py[i]); y1 = std::max(y1, T.py[i]);
                z0 = std::min(z0, T.pz[i]); z1 = std::max(z1, T.pz[i]);
            }
            const float bx = 0.5f * (x0 + x1), by = 0.5f * (y0 + y1), bz = 0.5f * (z0 + z1);
            float br2 = 0;
            for (int i = b; i < e; ++i) {
                if (T.pm[i] <= 0) continue;
                float dx = T.px[i] - bx, dy = T.py[i] - by, dz = T.pz[i] - bz;
                br2 = std::max(br2, dx * dx + dy * dy + dz * dz);
            }
            if (x0 > x1) continue;  // massless group
            const float br = std::sqrt(br2);

            // a single tree walk for the whole group
            L.n = 0;
            int i = 0;
            while (i < nn) {
                const Node& q = nd[i];
                float dx = q.cx - bx, dy = q.cy - by, dz = q.cz - bz;
                float t = q.open_r + br;
                if (dx * dx + dy * dy + dz * dz > t * t) {
                    L.push(q.cx, q.cy, q.cz, q.m);  // far enough: use the node's monopole
                    i = q.next;
                } else if (q.leaf) {
                    for (int j = q.start; j < q.start + q.count; ++j) L.push(T.px[j], T.py[j], T.pz[j], T.pm[j]);
                    i = q.next;
                } else {
                    i = i + 1;                      // open the node: first child
                }
            }

            // evaluate the list for every particle of the group (vectorised loop)
            const float* __restrict lx = L.x.data();
            const float* __restrict ly = L.y.data();
            const float* __restrict lz = L.z.data();
            const float* __restrict lm = L.m.data();
            const int m = L.n;
            for (int p = b; p < e; ++p) {
                if (active && !active[T.idx[p]]) continue;
                const float x = T.px[p], y = T.py[p], z = T.pz[p];
                float sx = 0, sy = 0, sz = 0;
#pragma omp simd reduction(+ : sx, sy, sz)
                for (int j = 0; j < m; ++j) {
                    float ex = lx[j] - x, ey = ly[j] - y, ez = lz[j] - z;
                    float r2 = ex * ex + ey * ey + ez * ez + eps2;   // Plummer softening
                    float inv = 1.0f / std::sqrt(r2);
                    float f = lm[j] * inv * inv * inv;
                    sx += f * ex; sy += f * ey; sz += f * ez;
                }
                const int k = T.idx[p];
                acc[3 * k] = sx;
                acc[3 * k + 1] = sy;
                acc[3 * k + 2] = sz;
            }
        }
    }
}

}  // namespace

extern "C" {

int nbody_num_threads() { return omp_get_max_threads(); }

// Barnes-Hut accelerations (G = 1) of the active particles (active[i] != 0, or all of them
// if active is NULL). The tree always contains every particle; inactive accelerations are
// left untouched. rebuild = 0 reuses (refits) the previous tree.
void nbody_accel_active(int n, const double* pos, const double* mass, double* acc, double theta, double eps,
                        const uint8_t* active, int rebuild) {
    compute_accel(n, pos, mass, acc, theta, eps, active, rebuild != 0);
}

// Exact O(N^2) accelerations, used by the tests to check Barnes-Hut.
void nbody_accel_direct(int n, const double* pos, const double* mass, double* acc, double eps) {
    const double eps2 = eps * eps;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        double ax = 0, ay = 0, az = 0;
        for (int j = 0; j < n; ++j) {
            double dx = pos[3 * j] - pos[3 * i], dy = pos[3 * j + 1] - pos[3 * i + 1], dz = pos[3 * j + 2] - pos[3 * i + 2];
            double r2 = dx * dx + dy * dy + dz * dz + eps2;
            double f = mass[j] / (r2 * std::sqrt(r2));
            ax += f * dx; ay += f * dy; az += f * dz;
        }
        acc[3 * i] = ax; acc[3 * i + 1] = ay; acc[3 * i + 2] = az;
    }
}

// Orthographic projection of the first n particles into img (ngroups, H, W) with a
// bilinear deposit. rot: 3x3 matrix whose rows are the screen x, y and depth axes.
void nbody_render(int n, const double* pos, const uint8_t* group, int ngroups,
                  const double* rot, const double* center, double scale,
                  int W, int H, float* img) {
    const size_t plane = size_t(W) * H;
    std::fill(img, img + ngroups * plane, 0.0f);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        double x = pos[3 * i] - center[0], y = pos[3 * i + 1] - center[1], z = pos[3 * i + 2] - center[2];
        double u = 0.5 * W + scale * (rot[0] * x + rot[1] * y + rot[2] * z) - 0.5;
        double v = 0.5 * H - scale * (rot[3] * x + rot[4] * y + rot[5] * z) - 0.5;
        if (!(u >= 0 && v >= 0 && u < W - 1 && v < H - 1)) continue;
        int iu = int(u), iv = int(v);
        float fu = float(u - iu), fv = float(v - iv);
        float* p = img + group[i] * plane + size_t(iv) * W + iu;
        // particles are spread out: few conflicts, so atomic adds are cheap
#pragma omp atomic
        p[0] += (1 - fu) * (1 - fv);
#pragma omp atomic
        p[1] += fu * (1 - fv);
#pragma omp atomic
        p[W] += (1 - fu) * fv;
#pragma omp atomic
        p[W + 1] += fu * fv;
    }
}

// Final 8-bit RGB image. For every group g (colour colors[g]):
//   l = log(1 + K img / ref) / log(1 + K) + glow   (glow upsampled bilinearly from gw x gh)
// then rgb = 255 (1 - exp(-exposure * sum_g l colors[g]))   (soft saturation to white).
void nbody_tonemap(int W, int H, int ngroups, const float* img, const float* glow, int gw, int gh,
                   const float* colors, float inv_ref, float K, float exposure, uint8_t* out) {
    const size_t plane = size_t(W) * H, gplane = size_t(gw) * gh;
    const float inv_logK = 1.0f / std::log1p(K);
    const float sx = float(gw) / W, sy = float(gh) / H;
#pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        float gyf = std::min(std::max((y + 0.5f) * sy - 0.5f, 0.0f), gh - 1.001f);
        int gy = int(gyf);
        float fy = gyf - gy;
        for (int x = 0; x < W; ++x) {
            float gxf = std::min(std::max((x + 0.5f) * sx - 0.5f, 0.0f), gw - 1.001f);
            int gx = int(gxf);
            float fx = gxf - gx;
            float r = 0, gr = 0, b = 0;
            for (int g = 0; g < ngroups; ++g) {
                float l = std::log1p(K * inv_ref * img[g * plane + size_t(y) * W + x]) * inv_logK;
                const float* q = glow + g * gplane + size_t(gy) * gw + gx;
                l += (1 - fy) * ((1 - fx) * q[0] + fx * q[1]) + fy * ((1 - fx) * q[gw] + fx * q[gw + 1]);
                r += l * colors[3 * g];
                gr += l * colors[3 * g + 1];
                b += l * colors[3 * g + 2];
            }
            uint8_t* o = out + 3 * (size_t(y) * W + x);
            o[0] = uint8_t(255.0f * (1 - std::exp(-exposure * r)) + 0.5f);
            o[1] = uint8_t(255.0f * (1 - std::exp(-exposure * gr)) + 0.5f);
            o[2] = uint8_t(255.0f * (1 - std::exp(-exposure * b)) + 0.5f);
        }
    }
}

}  // extern "C"
