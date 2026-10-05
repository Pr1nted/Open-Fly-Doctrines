/* Does the C brain give the SAME window as the JavaScript one?
 *
 *   cc -O2 -o /tmp/flyvalidate tools/validate.c src/fly_brain.c -Isrc
 *   /tmp/flyvalidate data/brain.odfly
 *
 * A port of a spiking simulator that is subtly wrong still produces plausible
 * spikes -- that is exactly why this compares counts per neuron against a
 * reference rather than eyeballing a total. Open-Fly keeps
 * tools/validate_brain.mjs for the same reason one level up, against Brian2.
 *
 * The comparison is bit-exact by construction: same float64 state, same
 * mulberry32 draws in the same order, same order of operations per step. If it
 * is not exact, something in the port is wrong, and "close" would be the worst
 * possible result to accept.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fly_brain.h"

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: validate <brain.odfly>\n"); return 2; }

    FILE *f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
    fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *file = (uint8_t *)malloc((size_t)len);
    if (fread(file, 1, (size_t)len, f) != (size_t)len) { fprintf(stderr, "short read\n"); return 2; }
    fclose(f);

    /* The same 64 MiB a mod gets, so this measures what the mod will have. */
    uint64_t cap = 64ull * 1024 * 1024;
    FlyArena arena = { (uint8_t *)malloc((size_t)cap), 0, cap };

    FlyBrain b;
    int rc = fly_open(&b, file, (uint64_t)len, &arena);
    if (rc != 0) { fprintf(stderr, "fly_open failed: %d\n", rc); return 1; }

    printf("  neurons %u  synapses %u  dn %u\n", b.n, b.nsyn, b.dn_count);
    printf("  dt %.4f  window params: em %.17g  es %.17g  k %.17g\n", b.dt, b.em, b.es, b.k);
    printf("  delay_steps %d  rfc_steps %d\n", b.delay_steps, b.rfc_steps);

    /* bitter is deliberately zero: a channel at zero must draw NO random
     * numbers, and if the port got that wrong every later draw would shift. */
    double rates[FLY_CHANNELS] = { 50.0, 0.0, 10.0, 100.0 };
    uint16_t *counts = (uint16_t *)malloc((size_t)b.n * 2);

    /* Unbounded: this is the run that must match the JavaScript exactly, and a
     * budget that cut it short would compare two different windows. */
    fly_run_window(&b, rates, 200.0, 20240922u, counts, 0);

    unsigned long total = 0, active = 0, dn_total = 0;
    unsigned long long hash = 14695981039346656037ull;   /* 0xcbf29ce484222325 */
    for (uint32_t i = 0; i < b.n; i++) {
        if (counts[i]) { total += counts[i]; active++; }
        /* FNV-1a over (index, count) for every spiking neuron: a total alone
         * would match even if the spikes were in the wrong places. */
        if (counts[i]) {
            unsigned char bytes[6];
            bytes[0] = (unsigned char)(i); bytes[1] = (unsigned char)(i >> 8);
            bytes[2] = (unsigned char)(i >> 16); bytes[3] = (unsigned char)(i >> 24);
            bytes[4] = (unsigned char)(counts[i]); bytes[5] = (unsigned char)(counts[i] >> 8);
            for (int k = 0; k < 6; k++) { hash ^= bytes[k]; hash *= 1099511628211ull; }
        }
    }
    for (uint32_t j = 0; j < b.dn_count; j++) dn_total += counts[b.dn[j]];

    printf("  spikes %lu  neurons that fired %lu  descending spikes %lu\n",
           total, active, dn_total);
    printf("  fingerprint %016llx\n", hash);
    printf("  arena used %.1f MB of %.0f MB\n", (double)arena.used / 1e6, (double)cap / 1e6);
    return 0;
}
