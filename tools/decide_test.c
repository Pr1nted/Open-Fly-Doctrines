/* Does the fly actually CHOOSE?
 *
 *   cc -O2 -o /tmp/flydecide tools/decide_test.c src/mod.c src/fly_brain.c \
 *      -Isrc -I$OD/sdk -DGEARBOX_IMPORT\(m,n\)= && /tmp/flydecide
 *
 * odmod-check proves the mod LOADS: the asset is read, the connectome parses,
 * the simulator fits. It never calls mod_ai_choose, so none of the deciding is
 * exercised by it -- the senses, the window, the group scoring and the legality
 * handling would all have been untested code shipped inside a working mod.
 *
 * This stubs the host and drives real decisions: a position, a legality mask,
 * and an answer that has to be one of the legal moves.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* The block mod.c allocates from when it is not a wasm module. */
uint8_t *fly_test_heap;
uint64_t fly_test_heap_bytes;

/* ---- the fake world ------------------------------------------------------ */

static int      g_turn = 5;
static uint32_t g_provinces_owned = 9;
static double   g_gross = 100.0, g_net = 12.0, g_treasury = 240.0;
static int      g_wars[4] = {0, 0, 0, 0};
static uint32_t g_actions = 12;
static char     g_legal[64];

void gearbox_log(int level, const char *msg, uint32_t len) {
    (void)level; printf("      [mod] %.*s\n", (int)len, msg);
}
uint32_t gearbox_asset_size(const char *name, uint32_t len) {
    (void)name; (void)len;
    FILE *f = fopen("data/brain.odfly", "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END); long n = ftell(f); fclose(f);
    return (uint32_t)n;
}
uint32_t gearbox_asset_read(const char *name, uint32_t len, void *buf, uint32_t cap) {
    (void)name; (void)len;
    FILE *f = fopen("data/brain.odfly", "rb");
    if (!f) return 0;
    size_t got = fread(buf, 1, cap, f);
    fclose(f);
    return (uint32_t)got;
}
uint32_t gearbox_turn_number(void)                    { return (uint32_t)g_turn; }
uint32_t gearbox_country_count(void)                  { return 4; }
uint32_t gearbox_country_at(uint32_t i)               { return i; }
uint32_t gearbox_country_is_ai(uint32_t c)            { (void)c; return 1; }
double   gearbox_country_treasury(uint32_t c)         { (void)c; return g_treasury; }
uint32_t gearbox_country_province_count(uint32_t c)   { (void)c; return g_provinces_owned; }
uint32_t gearbox_province_count(void)                 { return 900; }
double   gearbox_country_income_gross(uint32_t c)     { (void)c; return g_gross; }
double   gearbox_country_income_net(uint32_t c)       { (void)c; return g_net; }
uint32_t gearbox_at_war(uint32_t a, uint32_t b)       { (void)a; return b < 4 ? (uint32_t)g_wars[b] : 0; }
uint32_t gearbox_action_count(uint32_t module)        { (void)module; return g_actions; }
uint32_t gearbox_action_valid(uint32_t module, char *buf, uint32_t cap) {
    (void)module;
    uint32_t n = g_actions < cap ? g_actions : cap;
    memcpy(buf, g_legal, n);
    return g_actions;
}

int32_t  mod_load(void);
uint32_t mod_ai_choose(uint32_t country, uint32_t module);

#define INVALID 0xFFFFFFFFu

static int check(const char *what, int ok) {
    printf("  %-58s %s\n", what, ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}

int main(void) {
    fly_test_heap_bytes = 64ull * 1024 * 1024;
    fly_test_heap = (uint8_t *)malloc((size_t)fly_test_heap_bytes);

    if (mod_load() != 0) { printf("  mod_load FAILED\n"); return 1; }
    int bad = 0;

    memset(g_legal, 1, sizeof g_legal);

    /* THE FIRST WINDOW DECIDES NOTHING, and that is the design rather than a
     * wrinkle: the decoder scores how much each neuron fired compared with its
     * own recent baseline, and the first window is what establishes that
     * baseline. ParrotZero plays that turn. */
    bad += check("the first window sets the baseline and defers",
                 mod_ai_choose(0, 0) == INVALID);

    g_turn = 6;
    uint32_t a = mod_ai_choose(0, 0);
    bad += check("a decision comes back once it is warm", a != INVALID);
    bad += check("and it is within the action space", a < g_actions);

    /* Same position, same turn, same module: a replay must replay. */
    uint32_t again = mod_ai_choose(0, 0);
    bad += check("the same position gives the same answer", again == a);

    /* Other modules read the SAME window rather than running the brain again. */
    uint32_t m1 = mod_ai_choose(0, 1), m2 = mod_ai_choose(0, 2);
    bad += check("other modules answer from the same window",
                 m1 < g_actions && m2 < g_actions);

    /* Only one action legal: it must be that one, and the brain must not run. */
    memset(g_legal, 0, sizeof g_legal);
    g_legal[7] = 1;
    g_turn = 7;
    uint32_t forced = mod_ai_choose(0, 0);
    bad += check("one legal move is taken without waking the fly", forced == 7);

    /* A mask that forbids everything is not a crash, it is no opinion. */
    memset(g_legal, 0, sizeof g_legal);
    g_turn = 8;
    bad += check("no legal move means no opinion", mod_ai_choose(0, 0) == INVALID);

    /* A country that is not the pinned seat is declined, EVERY time -- and the
     * seat answers often, but not always: a turn on which every group fired
     * less than usual is a turn the fly has no opinion about, and saying so
     * lets ParrotZero decide instead of inventing a preference. */
    memset(g_legal, 1, sizeof g_legal);
    int other_answered = 0, seat_answered = 0;
    for (int t = 30; t < 40; t++) {
        g_turn = t;
        g_provinces_owned = (uint32_t)(9 + (t % 5) * 4);
        if (mod_ai_choose(3, 0) != INVALID) other_answered++;
        if (mod_ai_choose(0, 0) != INVALID) seat_answered++;
    }
    bad += check("a country that is not the seat is never answered", other_answered == 0);
    bad += check("the seat answers on most turns", seat_answered >= 5);

    /* A different position must be able to give a different answer -- otherwise
     * the senses are not reaching the brain at all, which would look exactly
     * like a working mod that ignores the game. */
    extern double fly_debug_rates[4];
    int seen_change = 0;
    uint32_t base = mod_ai_choose(0, 0);
    printf("      turn  sugar bitter  water    jon   ->  action\n");
    for (int t = 9; t < 24; t++) {
        g_turn = t;
        g_provinces_owned = (uint32_t)(9 + (t - 8) * 3);   /* winning land */
        g_wars[1] = (t % 3 == 0);                          /* and wars come and go */
        g_net = (t % 2) ? 40.0 : -25.0;
        uint32_t got = mod_ai_choose(0, 0);
        extern double fly_debug_dn_total, fly_debug_score[16], fly_debug_size[16];
        printf("      %4d %6.1f %6.1f %6.1f %6.1f  dn=%5.0f  g0..3=%.3f/%.3f/%.3f/%.3f (sizes %.0f/%.0f/%.0f/%.0f)  -> %u\n",
               t, fly_debug_rates[0], fly_debug_rates[1], fly_debug_rates[2], fly_debug_rates[3],
               fly_debug_dn_total,
               fly_debug_score[0], fly_debug_score[1], fly_debug_score[2], fly_debug_score[3],
               fly_debug_size[0], fly_debug_size[1], fly_debug_size[2], fly_debug_size[3], got);
        if (got != base) seen_change = 1;
    }
    bad += check("the position changes the answer", seen_change);

    printf("\n  %s\n", bad ? "SOMETHING IS WRONG" : "the fly decides");
    return bad ? 1 : 0;
}
