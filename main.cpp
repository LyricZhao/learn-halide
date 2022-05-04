#include <cstdio>

#include <Halide.h>
#include <halide_image_io.h>


class SharpenPipeline {
private:
    Halide::Var x, y, c, i, ii, xo, yo, xi, yi;

public:
    Halide::Buffer<uint8_t> input;
    Halide::Func lut, padded, padded_16, sharpen, curved;

    explicit SharpenPipeline(const Halide::Buffer<uint8_t>& in): input(in) {
        lut(i) = Halide::cast<uint8_t>(
                Halide::clamp(Halide::pow(i / 255.0f, 1.2f) * 255.0f, 0, 255));
        padded(x, y, c) = input(Halide::clamp(x, 0, input.width() - 1),
                                Halide::clamp(y, 0, input.height() - 1),
                                c);
        padded_16(x, y, c) = Halide::cast<uint16_t>(padded(x, y, c));
        sharpen(x, y, c) = (padded_16(x, y, c) * 2 -
                            (padded_16(x - 1, y, c) +
                             padded_16(x, y - 1, c) +
                             padded_16(x + 1, y, c) +
                             padded_16(x, y + 1, c)) /
                            4);
        curved(x, y, c) = lut(sharpen(x, y, c));
    }

    void ScheduleForCPU() {
        lut.compute_root();
        curved.reorder(c, x, y).bound(c, 0, 3).unroll(c);
        curved.split(y, yo, yi, 16).parallel(yo);
        sharpen.compute_at(curved, yi);
        sharpen.vectorize(x, 8);
        padded.store_at(curved, yo).compute_at(curved, yi);
        padded.vectorize(x, 16);

        Halide::Target target = Halide::get_host_target();
        curved.compile_jit(target);
    }
};


int main(int argc, char **argv) {
    Halide::Buffer<uint8_t> input = Halide::Tools::load_image("examples/tree.jpeg");
    Halide::Buffer<uint8_t> output(input.width(), input.height(), input.channels());

    printf("Running pipeline on CPU ...\n");
    SharpenPipeline pipeline(input);
    pipeline.ScheduleForCPU();
    pipeline.curved.realize(output);

    return 0;
}
