#include <jack/jack.h>
#include <fftw3.h>
#include <atomic>
#include <complex>
#include <cstring>
#include <cmath>
#include <unistd.h>

typedef std::complex<float> cfloat;

namespace {
    jack_client_t *client = nullptr;
    jack_port_t *port_in = nullptr;
    jack_port_t *port_out = nullptr;
    std::atomic<int> measure_started(0);
    std::atomic<int> measure_finished(0);

    //
    constexpr unsigned fft_size = 2048;
    constexpr unsigned fft_nbins = fft_size / 2 + 1;

    //
    constexpr float silence_threshold = 1e-2f;  // -40 dB
}

struct Measure_Context {
    float *fft_in = nullptr;
    cfloat *fft_out = nullptr;
    fftwf_plan fft_plan = nullptr;

    unsigned current_bin = ~0u;
    unsigned fft_in_fill = 0;
    float oscillator_phase = 0;
    cfloat *response = nullptr;

    unsigned silence_frames_count = 0;
    unsigned silence_frames_needed = 0;
};

static int process(unsigned nframes, void *userdata)
{
    Measure_Context &ctx = *(Measure_Context *)userdata;
    const float *input = (float *)jack_port_get_buffer(::port_in, nframes);
    float *output = (float *)jack_port_get_buffer(::port_out, nframes);

    std::memset(output, 0, nframes * sizeof(float));

    if (!measure_started.load())
        return 0;

    for (unsigned i = 0; i < nframes; ++i)
        ctx.silence_frames_count = (std::abs(input[i]) < silence_threshold) ?
            (ctx.silence_frames_count + 1) : 0;

    if (ctx.current_bin == fft_nbins) {
        measure_finished.store(1);
        std::atomic_thread_fence(std::memory_order_release);
        return 0;
    }

    if (ctx.current_bin != ~0u && ctx.fft_in_fill == fft_size) {
        for (unsigned i = 0; i < fft_size; ++i) {
            double k = i / (double)(fft_size - 1);
            //ctx.fft_in[i] *= 0.5 * (1.0 - std::cos(2 * M_PI * k));
            ctx.fft_in[i] *= 0.42 - 0.5 * std::cos(2 * M_PI * k) + 0.08 * std::cos(4 * M_PI * k);
        }
        fftwf_execute(ctx.fft_plan);
        ctx.response[ctx.current_bin] = ctx.fft_out[ctx.current_bin];
    }

    bool change_bin = ctx.current_bin == ~0u || ctx.fft_in_fill == fft_size;
    if (change_bin) {
        if (ctx.silence_frames_count < ctx.silence_frames_needed)
            return 0;
        ++ctx.current_bin;
        ctx.fft_in_fill = 0;
        ctx.oscillator_phase = 0;
    }

    float oscillator_freq = (float)ctx.current_bin / fft_size;
    for (unsigned i = 0; i < nframes && ctx.fft_in_fill < fft_size; ++i) {
        output[i] = std::sin(ctx.oscillator_phase);
        ctx.oscillator_phase += oscillator_freq;
        ctx.oscillator_phase -= (int)ctx.oscillator_phase;
    }

    bool record_input = !change_bin;
    if (record_input) {
        for (unsigned i = 0; i < nframes && ctx.fft_in_fill < fft_size; ++i)
            ctx.fft_in[ctx.fft_in_fill++] = input[i];
    }

    return 0;
}

static void setup_fft(Measure_Context &ctx)
{
    float *fft_in = ctx.fft_in = fftwf_alloc_real(fft_size);
    cfloat *fft_out = ctx.fft_out = (cfloat *)fftwf_alloc_complex(fft_nbins);
    ctx.fft_plan = fftwf_plan_dft_r2c_1d(fft_size, fft_in, (fftwf_complex *)fft_out, FFTW_MEASURE);
    ctx.response = new cfloat[fft_nbins];
}

static bool setup_jack(Measure_Context &ctx)
{
    jack_client_t *client = jack_client_open("MeasureResponse", JackNoStartServer, nullptr);
    if (!client) {
        fprintf(stderr, "Could not open a JACK client.\n");
        return false;
    }
    ::client = client;

    port_in = jack_port_register(client, "Analyzer In", JACK_DEFAULT_AUDIO_TYPE, JackPortIsInput, 0);
    port_out = jack_port_register(client, "Generator Out", JACK_DEFAULT_AUDIO_TYPE, JackPortIsOutput, 0);
    if (!port_in || !port_out) {
        fprintf(stderr, "Could not register JACK ports.\n");
        return false;
    }

    jack_set_process_callback(client, &process, &ctx);
    return true;
}

int main()
{
    Measure_Context ctx;

    if (!setup_jack(ctx))
        return 1;

    setup_fft(ctx);

    jack_client_t *client = ::client;

    float sample_rate = jack_get_sample_rate(client);
    ctx.silence_frames_needed = std::ceil(10e-3 * sample_rate);

    jack_activate(client);

    const char *filename = "response.dat";

    printf(">> About to measure frequency response\n");
    printf(">> Data will be saved to '%s'\n", filename);
    printf(">> * Connect ports in JACK.\n");
    printf(">> * Press ENTER when ready.\n");
    printf("...");
    fflush(stdout);
    getchar();
    printf(">> Start!\n");

    measure_started.store(1);
    while (measure_finished.load() == 0) {
        if (ctx.current_bin != ~0u) {
            printf("%*s\r", 80, "");
            printf("Progress %u/%u", ctx.current_bin + 1, fft_nbins);
        }
        // sleep(1);
        timespec ts;
        ts.tv_sec = 0;
        ts.tv_nsec = 100 * 1000 * 1000;
        nanosleep(&ts, nullptr);
    }
    printf("\n");
    std::atomic_thread_fence(std::memory_order_acquire);

    float fs = jack_get_sample_rate(client);
    float fnyquist = 0.5f * fs;

    FILE *fh = fopen(filename, "w");
    if (!fh) {
        fprintf(stderr, "Could not open the file for writing.\n");
        return 1;
    }

    for (unsigned i = 0; i < fft_nbins; ++i) {
        float freq = i * (fs / fft_size);
        float amp = std::abs(ctx.response[i]) / fft_size;
        float phase = std::arg(ctx.response[i]);
        fprintf(fh, "%f %f %f\n", freq, amp, phase);
    }

    if (fflush(fh) != 0) {
        fprintf(stderr, "Could not write the result file.\n");
        return 1;
    }

    fclose(fh);

    return 0;
}
