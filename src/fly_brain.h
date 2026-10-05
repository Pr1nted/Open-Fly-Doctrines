/* The FlyWire whole-brain model, in C, for a Gearbox mod.
 *
 * A port of Open-Fly's web/brain.js, which is itself the model brain.py runs in
 * Brian2 -- Shiu et al., Nature 2024. Same equations, same constants, and the
 * same ORDER OF OPERATIONS per step, because the order is load-bearing:
 *
 *   dv/dt = (v_0 - v + g) / t_mbr     (frozen while refractory)
 *   dg/dt = -g / tau                   (frozen while refractory)
 *   spike when v > v_th; then v = v_rst, g = 0, refractory for t_rfc
 *   a spike adds w = count * w_syn to g of each target, t_dly later
 *   a sensory input spike adds w_syn * f_poi to v of its neuron at once
 *
 * DELIBERATELY FREE OF GEARBOX. The simulator knows nothing about the game, so
 * it compiles natively for tools/validate.c and to wasm for the mod from the
 * same source. A port validated in one build and shipped from another would be
 * two ports.
 *
 * FLOAT64 THROUGHOUT, and that is not a default. Open-Fly found that 32-bit
 * state agreed with Brian2 on every mechanism and still diverged after ~55 ms
 * of stimulus, rounding near threshold. The cost is ~2 MB.
 *
 * EVENT-DRIVEN. Of 138,639 neurons a 200 ms window moves a few thousand off
 * rest, and a neuron at rest stays at rest under these equations -- so only
 * neurons that have received input are integrated. That is what makes 2,000
 * steps over a whole brain affordable inside a per-decision fuel budget.
 */
#ifndef FLY_BRAIN_H
#define FLY_BRAIN_H

#include <stdint.h>

#define FLY_CHANNELS 4                 /* sugar, bitter, water, jon */

/* A bump allocator. There is no libc in a freestanding wasm module, and a mod
 * that needed one would be carrying an allocator to make exactly one large
 * allocation per lifetime. */
typedef struct {
    uint8_t *base;
    uint64_t used, cap;
} FlyArena;

void *fly_alloc(FlyArena *a, uint64_t bytes, uint64_t align);

/* Spikes in flight: one bucket per step of axonal delay. Grown on demand rather
 * than preallocated -- brain.js reserves 65,536 entries per bucket, which for
 * 19 buckets is 14.9 MB we cannot spare next to a 42.8 MB connectome, and the
 * measured traffic is ~137 deliveries a step. */
typedef struct {
    int32_t  *post;
    double   *w;
    uint32_t  len, cap;
} FlyBucket;

typedef struct {
    uint32_t n, nsyn;

    /* Parameters, read from the packed file rather than compiled in, so the
     * mod cannot drift from the brain it was packed against. */
    double v_0, v_rst, v_th, t_mbr, tau, t_rfc, t_dly, w_syn, f_poi, dt;

    /* Derived once. */
    double em, es, k, w_poisson;
    int32_t delay_steps, rfc_steps;

    const uint32_t *row_off;           /* n + 1 offsets into blob */
    const uint8_t  *blob;              /* varint (delta, zigzag weight) pairs */

    uint32_t        sensory_count[FLY_CHANNELS];
    const uint32_t *sensory[FLY_CHANNELS];
    uint32_t        dn_count;
    const uint32_t *dn;
    uint32_t        partition_seed;

    /* State. */
    double   *v, *g;
    uint16_t *rfc;
    uint8_t  *no_rfc, *awake;
    int32_t  *awake_list;
    uint32_t  awake_count;
    int32_t  *spikes_step;
    FlyBucket *ring;
    int32_t   ring_len, head;

    FlyArena *arena;
} FlyBrain;

/* Parse the packed file and allocate state. Returns 0 on success, or a negative
 * code: -1 bad magic, -2 truncated, -3 out of arena. */
int fly_open(FlyBrain *b, const uint8_t *file, uint64_t len, FlyArena *arena);

/* One decision window. `rates` is Hz per channel, `counts` collects spikes per
 * neuron (n entries, zeroed by the callee).
 *
 * `max_integrations` bounds the thinking; 0 means unbounded, which is what the
 * JavaScript does and what tools/validate.c must use to compare against it.
 *
 * WHY A BOUND IS NEEDED AT ALL. A window's cost depends on the POSITION: a fly
 * with four senses firing wakes far more neurons than a quiet one. A quiet turn
 * measures ~700M-1,000M WAMR instructions; a busy one exceeds 2,000,000,000 --
 * past the largest budget a Gearbox mod can be given, because the runtime's
 * instruction limit is an int and INT32_MAX is the wall. Without a bound the
 * fly traps on its first eventful turn, and a trap does not merely lose that
 * decision: the host disables the mod.
 *
 * So the fly gets as much of the window as its budget allows, and on a busy
 * turn it thinks for less of it -- which is a fair description of a small brain
 * under load. Returns the number of steps actually run. */
uint32_t fly_run_window(FlyBrain *b, const double rates[FLY_CHANNELS],
                        double window_ms, uint32_t seed, uint16_t *counts,
                        uint64_t max_integrations);

/* mulberry32, exactly as web/decide.js draws it. Exposed because the draw
 * ORDER is part of the model: the same seed must give the same window. */
uint32_t fly_rand_u32(uint32_t *state);

#endif /* FLY_BRAIN_H */
