#include "fly_brain.h"

/* No libc: these are the only three things the simulator needs from one. */
static void fly_zero(void *p, uint64_t n) {
    uint8_t *q = (uint8_t *)p;
    for (uint64_t i = 0; i < n; i++) q[i] = 0;
}
static void fly_copy(void *dst, const void *src, uint64_t n) {
    uint8_t *d = (uint8_t *)dst; const uint8_t *s = (const uint8_t *)src;
    for (uint64_t i = 0; i < n; i++) d[i] = s[i];
}
static double fly_fabs(double x) { return x < 0 ? -x : x; }

/* exp(), for the two decay constants. Computed ONCE at open, never per step,
 * so a compact implementation costs nothing at run time. Range reduction to
 * [-ln2/2, ln2/2] then a Taylor series: the arguments here are around
 * -0.005 and -0.02, where this is accurate to well past double precision. */
static double fly_exp(double x) {
    const double LN2 = 0.6931471805599453;
    int k = (int)(x / LN2 + (x < 0 ? -0.5 : 0.5));
    double r = x - (double)k * LN2, term = 1.0, sum = 1.0;
    for (int i = 1; i < 18; i++) { term *= r / (double)i; sum += term; }
    double p = 1.0;
    if (k > 0) { for (int i = 0; i < k; i++) p *= 2.0; }
    else       { for (int i = 0; i < -k; i++) p *= 0.5; }
    return sum * p;
}

void *fly_alloc(FlyArena *a, uint64_t bytes, uint64_t align) {
    uint64_t p = (a->used + (align - 1)) & ~(align - 1);
    if (p + bytes > a->cap) return 0;
    a->used = p + bytes;
    return a->base + p;
}

/* mulberry32. Every operation is 32-bit wrapping, which is what Math.imul does
 * and what uint32_t arithmetic does here -- so the two agree bit for bit. */
uint32_t fly_rand_u32(uint32_t *s) {
    *s = *s + 0x6d2b79f5u;
    uint32_t t = *s;
    t = (t ^ (t >> 15)) * (1u | t);
    t = (t + ((t ^ (t >> 7)) * (61u | t))) ^ t;
    return t ^ (t >> 14);
}
static double fly_rand(uint32_t *s) {
    return (double)fly_rand_u32(s) / 4294967296.0;
}

/* ------------------------------------------------------------- the file ---- */

static uint32_t rd_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static double rd_f64(const uint8_t *p) {
    uint64_t u = 0;
    for (int i = 7; i >= 0; i--) u = (u << 8) | p[i];
    double d; fly_copy(&d, &u, 8); return d;
}

int fly_open(FlyBrain *b, const uint8_t *f, uint64_t len, FlyArena *arena) {
    static const char kMagic[8] = {'O','D','F','L','Y','1','\0','\0'};
    if (len < 16) return -2;
    for (int i = 0; i < 8; i++) if ((char)f[i] != kMagic[i]) return -1;

    fly_zero(b, sizeof *b);
    b->arena = arena;
    uint64_t o = 8;
    b->n = rd_u32(f + o); o += 4;
    b->nsyn = rd_u32(f + o); o += 4;

    const double *pp = 0; (void)pp;
    b->v_0   = rd_f64(f + o + 0 * 8);
    b->v_rst = rd_f64(f + o + 1 * 8);
    b->v_th  = rd_f64(f + o + 2 * 8);
    b->t_mbr = rd_f64(f + o + 3 * 8);
    b->tau   = rd_f64(f + o + 4 * 8);
    b->t_rfc = rd_f64(f + o + 5 * 8);
    b->t_dly = rd_f64(f + o + 6 * 8);
    b->w_syn = rd_f64(f + o + 7 * 8);
    b->f_poi = rd_f64(f + o + 8 * 8);
    b->dt    = rd_f64(f + o + 9 * 8);
    o += 80;

    for (int c = 0; c < FLY_CHANNELS; c++) {
        if (o + 4 > len) return -2;
        b->sensory_count[c] = rd_u32(f + o); o += 4;
        b->sensory[c] = (const uint32_t *)(f + o);
        o += (uint64_t)b->sensory_count[c] * 4;
    }
    if (o + 4 > len) return -2;
    b->dn_count = rd_u32(f + o); o += 4;
    b->dn = (const uint32_t *)(f + o);
    o += (uint64_t)b->dn_count * 4;
    if (o + 12 > len) return -2;
    b->partition_seed = rd_u32(f + o); o += 4;
    uint64_t blob_len = 0;
    for (int i = 7; i >= 0; i--) blob_len = (blob_len << 8) | f[o + i];
    o += 8;

    b->row_off = (const uint32_t *)(f + o);
    o += (uint64_t)(b->n + 1) * 4;
    if (o + blob_len > len) return -2;
    b->blob = f + o;

    /* Derived. Rounding matches JS Math.round, which is floor(x + 0.5). */
    b->em = fly_exp(-b->dt / b->t_mbr);
    b->es = fly_exp(-b->dt / b->tau);
    b->k  = b->tau / (b->tau - b->t_mbr);
    b->w_poisson = b->w_syn * b->f_poi;
    b->delay_steps = (int32_t)(b->t_dly / b->dt + 0.5);
    b->rfc_steps   = (int32_t)(b->t_rfc / b->dt + 0.5);

    const uint32_t n = b->n;
    b->v          = (double   *)fly_alloc(arena, (uint64_t)n * 8, 8);
    b->g          = (double   *)fly_alloc(arena, (uint64_t)n * 8, 8);
    b->rfc        = (uint16_t *)fly_alloc(arena, (uint64_t)n * 2, 2);
    b->no_rfc     = (uint8_t  *)fly_alloc(arena, n, 1);
    b->awake      = (uint8_t  *)fly_alloc(arena, n, 1);
    b->awake_list = (int32_t  *)fly_alloc(arena, (uint64_t)n * 4, 4);
    b->spikes_step= (int32_t  *)fly_alloc(arena, (uint64_t)n * 4, 4);
    if (!b->v || !b->g || !b->rfc || !b->no_rfc || !b->awake ||
        !b->awake_list || !b->spikes_step) return -3;

    b->ring_len = b->delay_steps + 1;
    b->ring = (FlyBucket *)fly_alloc(arena, (uint64_t)b->ring_len * sizeof(FlyBucket), 8);
    if (!b->ring) return -3;
    for (int i = 0; i < b->ring_len; i++) {
        /* 4096, not brain.js's 65536: measured traffic is ~137 deliveries a
         * step, and 19 buckets at 65536 would be 14.9 MB beside a 42.8 MB
         * connectome. Doubled on demand below, so a burst still fits. */
        b->ring[i].cap = 4096;
        b->ring[i].len = 0;
        b->ring[i].post = (int32_t *)fly_alloc(arena, 4096 * 4, 4);
        b->ring[i].w    = (double  *)fly_alloc(arena, 4096 * 8, 8);
        if (!b->ring[i].post || !b->ring[i].w) return -3;
    }

    for (uint32_t i = 0; i < n; i++) { b->v[i] = b->v_0; b->g[i] = 0.0; }
    fly_zero(b->rfc, (uint64_t)n * 2);
    fly_zero(b->no_rfc, n);
    fly_zero(b->awake, n);
    b->awake_count = 0;
    b->head = 0;
    for (int c = 0; c < FLY_CHANNELS; c++)
        for (uint32_t i = 0; i < b->sensory_count[c]; i++)
            b->no_rfc[b->sensory[c][i]] = 1;
    return 0;
}

/* --------------------------------------------------------------- the step -- */

static void wake(FlyBrain *b, int32_t i) {
    if (!b->awake[i]) { b->awake[i] = 1; b->awake_list[b->awake_count++] = i; }
}

static int push(FlyBrain *b, FlyBucket *k, int32_t target, double w) {
    if (k->len == k->cap) {
        uint32_t cap2 = k->cap * 2;
        int32_t *p = (int32_t *)fly_alloc(b->arena, (uint64_t)cap2 * 4, 4);
        double  *q = (double  *)fly_alloc(b->arena, (uint64_t)cap2 * 8, 8);
        if (!p || !q) return 0;        /* arena exhausted: drop, never corrupt */
        fly_copy(p, k->post, (uint64_t)k->len * 4);
        fly_copy(q, k->w,    (uint64_t)k->len * 8);
        k->post = p; k->w = q; k->cap = cap2;
    }
    k->post[k->len] = target; k->w[k->len] = w; k->len++;
    return 1;
}

static uint64_t g_integrations;

static void step(FlyBrain *b, const double *rates, uint32_t *rng, uint16_t *counts) {
    const double v0 = b->v_0, vth = b->v_th, em = b->em, es = b->es, kk = b->k;
    double *v = b->v, *g = b->g;
    uint16_t *rfc = b->rfc;
    int32_t *list = b->awake_list, *spk = b->spikes_step;

    /* groups: integrate every awake, non-refractory neuron; retire the settled */
    uint32_t live = 0, n_spk = 0;
    g_integrations += b->awake_count;
    for (uint32_t j = 0; j < b->awake_count; j++) {
        int32_t i = list[j];
        if (rfc[i] > 0) { list[live++] = i; continue; }
        double gi = g[i];
        double vi = v0 + (v[i] - v0 - kk * gi) * em + kk * gi * es;
        double gn = gi * es;
        v[i] = vi; g[i] = gn;
        if (vi > vth) spk[n_spk++] = i;
        if (fly_fabs(vi - v0) < 1e-5 && fly_fabs(gn) < 1e-6) {
            v[i] = v0; g[i] = 0.0; b->awake[i] = 0;
        } else {
            list[live++] = i;
        }
    }
    b->awake_count = live;

    /* Delayed deliveries land on g, input spikes on v -- both before the reset
     * below, as in Brian2.
     *
     * A REFRACTORY TARGET IGNORES THE SPIKE. g is declared "(unless
     * refractory)", and in Brian2 that guards synaptic updates to it as well as
     * its integration: an input landing inside t_rfc is lost, not stored for
     * later. Storing it instead let a neuron fire one input sooner and
     * compounded into ~25% more spikes downstream of a strong stimulus. */
    FlyBucket *bucket = &b->ring[b->head];
    for (uint32_t s = 0; s < bucket->len; s++) {
        int32_t t = bucket->post[s];
        if (rfc[t] > 0) continue;
        g[t] += bucket->w[s];
        wake(b, t);
    }
    bucket->len = 0;

    /* Channel order is sugar, bitter, water, jon -- the order brain.json lists
     * them and therefore the order `for (const ch in this.sensory)` walks them.
     * It decides the sequence of random draws, so it is part of the model. A
     * channel at zero draws NOTHING, which is why the skip is before the loop. */
    for (int c = 0; c < FLY_CHANNELS; c++) {
        double r = rates[c];
        if (r <= 0.0) continue;
        double p = r * b->dt * 1e-3;
        const uint32_t *idx = b->sensory[c];
        for (uint32_t s = 0; s < b->sensory_count[c]; s++) {
            if (fly_rand(rng) < p) {
                int32_t t = (int32_t)idx[s];
                if (rfc[t] > 0) continue;
                v[t] += b->w_poisson;
                wake(b, t);
            }
        }
    }

    /* resets, and each spike queued t_dly ahead */
    FlyBucket *target = &b->ring[(b->head + b->delay_steps) % b->ring_len];
    for (uint32_t s = 0; s < n_spk; s++) {
        int32_t i = spk[s];
        v[i] = b->v_rst; g[i] = 0.0;
        /* Brian2 frees a neuron once timestep(t - lastspike) >= timestep(t_rfc):
         * after a spike at step k it is frozen for k+1 .. k+21 and integrates
         * again at k+22. The counter runs down at the END of each step,
         * including this one, so 22 here freezes exactly those 21. */
        rfc[i] = b->no_rfc[i] ? 0 : (uint16_t)b->rfc_steps;
        if (counts) counts[i]++;

        /* Walk this neuron's row straight out of the varint blob. Decoding on
         * demand is what lets the connectome stay compressed in memory: only
         * the rows of neurons that actually spike are ever expanded, and the
         * measurement says that is a few thousand of 138,639. */
        uint32_t off = b->row_off[i], end = b->row_off[i + 1];
        const uint8_t *p = b->blob;
        uint32_t prev = 0;
        /* Charged to the same budget as integration. A busy turn spikes far
         * more, and each spike decodes a whole row of synapses -- counting only
         * integrations left that work invisible, and the budget then failed to
         * bound exactly the turns it existed to bound. */
        g_integrations += (end - off) / 2;
        while (off < end) {
            uint32_t d = 0, sh = 0, byte;
            do { byte = p[off++]; d |= (byte & 0x7Fu) << sh; sh += 7; } while (byte & 0x80u);
            uint32_t z = 0; sh = 0;
            do { byte = p[off++]; z |= (byte & 0x7Fu) << sh; sh += 7; } while (byte & 0x80u);
            int32_t w = (int32_t)((z >> 1) ^ (uint32_t)(-(int32_t)(z & 1u)));
            prev += d;
            push(b, target, (int32_t)prev, (double)w * b->w_syn);
        }
    }

    /* a refractory neuron's clock runs down whether or not it was integrated */
    for (uint32_t j = 0; j < b->awake_count; j++) {
        int32_t i = list[j];
        if (rfc[i] > 0) rfc[i]--;
    }
    b->head = (b->head + 1) % b->ring_len;
}

uint32_t fly_run_window(FlyBrain *b, const double rates[FLY_CHANNELS],
                        double window_ms, uint32_t seed, uint16_t *counts,
                        uint64_t max_integrations) {
    uint32_t rng = seed;
    static const double kNoRates[FLY_CHANNELS] = {0, 0, 0, 0};
    g_integrations = 0;

    /* Flush, exactly as runWindow does: 2 ms of silence, then every awake
     * neuron is reset to rest before the window proper is counted. */
    int flush_steps = (int)(2.0 / b->dt + 0.5);
    for (int s = 0; s < flush_steps; s++) step(b, kNoRates, &rng, 0);
    for (uint32_t j = 0; j < b->awake_count; j++) {
        int32_t i = b->awake_list[j];
        b->v[i] = b->v_0; b->g[i] = 0.0;
    }

    if (counts) fly_zero(counts, (uint64_t)b->n * 2);
    int steps = (int)(window_ms / b->dt + 0.5);
    int s = 0;
    for (; s < steps; s++) {
        /* Checked BETWEEN steps, never inside one: a half-integrated step
         * leaves the state meaningless, and the state meaning something is the
         * whole point of this simulator. */
        if (max_integrations && g_integrations >= max_integrations) break;
        step(b, rates, &rng, counts);
    }
    return (uint32_t)s;
}
