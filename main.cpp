#include <cstdio>

#include <Halide.h>
#include <halide_benchmark.h>


class AutoScheduled: public Halide::Generator<AutoScheduled> {
private:
    Var x{"x"}, y{"y"}, c{"c"};
    Func gray, Iy, Ix, Ixx, Iyy, Ixy, Sxx, Syy, Sxy, det, trace, harris;

public:
    Input<Buffer<float>> input{"input", 3};
    Input<float> factor{"factor"};

    Output<Buffer<float>> output1{"output1", 2};
    Output<Buffer<float>> output2{"output2", 2};

    static Expr sum3x3(const Func& f, const Var& x, const Var& y) {
        return f(x - 1, y - 1) + f(x - 1, y) + f(x - 1, y + 1) +
               f(x, y - 1) + f(x, y) + f(x, y + 1) +
               f(x + 1, y - 1) + f(x + 1, y) + f(x + 1, y + 1);
    }

    void generate() {
        Func in_b = Halide::BoundaryConditions::repeat_edge(input);
        gray(x, y) = 0.299f * in_b(x, y, 0) + 0.587f * in_b(x, y, 1) + 0.114f * in_b(x, y, 2);

        Iy(x, y) = gray(x - 1, y - 1) * (-1.0f / 12) + gray(x - 1, y + 1) * (1.0f / 12) +
                   gray(x, y - 1) * (-2.0f / 12) + gray(x, y + 1) * (2.0f / 12) +
                   gray(x + 1, y - 1) * (-1.0f / 12) + gray(x + 1, y + 1) * (1.0f / 12);

        Ix(x, y) = gray(x - 1, y - 1) * (-1.0f / 12) + gray(x + 1, y - 1) * (1.0f / 12) +
                   gray(x - 1, y) * (-2.0f / 12) + gray(x + 1, y) * (2.0f / 12) +
                   gray(x - 1, y + 1) * (-1.0f / 12) + gray(x + 1, y + 1) * (1.0f / 12);

        Ixx(x, y) = Ix(x, y) * Ix(x, y);
        Iyy(x, y) = Iy(x, y) * Iy(x, y);
        Ixy(x, y) = Ix(x, y) * Iy(x, y);
        Sxx(x, y) = sum3x3(Ixx, x, y);
        Syy(x, y) = sum3x3(Iyy, x, y);
        Sxy(x, y) = sum3x3(Ixy, x, y);
        det(x, y) = Sxx(x, y) * Syy(x, y) - Sxy(x, y) * Sxy(x, y);
        trace(x, y) = Sxx(x, y) + Syy(x, y);
        harris(x, y) = det(x, y) - 0.04f * trace(x, y) * trace(x, y);
        output1(x, y) = harris(x + 2, y + 2);
        output2(x, y) = factor * harris(x + 2, y + 2);
    }

    void schedule() {
        if (auto_schedule) {
            input.set_estimates({{0, 1024}, {0, 1024}, {0, 3}});
            factor.set_estimate(2.0f);
            output1.set_estimates({{0, 1024}, {0, 1024}});
            output2.set_estimates({{0, 1024}, {0, 1024}});
        } else {
            gray.compute_root();
            Iy.compute_root();
            Ix.compute_root();
        }
    }
};

HALIDE_REGISTER_GENERATOR(AutoScheduled, auto_schedule_gen)


// int main(int argc, char **argv) {
//     Halide::Runtime::Buffer<float> input(1024, 1024, 3);
//     for (int c = 0; c < input.channels(); ++c) {
//         for (int y = 0; y < input.height(); ++y) {
//             for (int x = 0; x < input.width(); ++x) {
//                 input(x, y, c) = rand();
//             }
//         }
//     }
//
//     Halide::Runtime::Buffer<float> output1(1024, 1024);
//     Halide::Runtime::Buffer<float> output2(1024, 1024);
//
//     double auto_schedule_off = Halide::Tools::benchmark(2, 5, [&]() {
//         auto_schedule_false(input, 2.0f, output1, output2);
//     });
//     printf("Manual schedule: %gms\n", auto_schedule_off * 1e3);
//
//     double auto_schedule_on = Halide::Tools::benchmark(2, 5, [&]() {
//         auto_schedule_true(input, 2.0f, output1, output2);
//     });
//     printf("Auto schedule: %gms\n", auto_schedule_on * 1e3);
//
//     return 0;
// }
