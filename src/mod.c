/* Fly Doctrines: a fruit fly's brain chooses one country's moves.
 *
 * ParrotZero -- OpenDoctrines' own AI -- keeps playing everyone else. This mod
 * takes ONE country and hands its decisions to a simulation of the whole
 * FlyWire connectome: 138,639 neurons and 15,091,983 synapses, leaky
 * integrate-and-fire, the model from Shiu et al. (Nature, 2024).
 *
 * HOW A GAME POSITION REACHES A FLY
 *
 * The fly has four senses, so the position is reduced to four firing rates,
 * exactly as Open Fly does it:
 *
 *   sugar   land gained since last turn, and profit
 *   bitter  land lost, losses, and wars newly declared
 *   water   treasury measured in turns of reserve
 *   jon     how many wars it is in          (jon = Johnston's organ, hearing)
 *
 * 200 ms of brain time later, the 1,299 descending neurons -- the ones that in
 * a real fly carry commands from brain to body -- have spiked some number of
 * times each. Those are dealt into as many groups as the decision has legal
 * moves, and the group that fired hardest picks the move.
 *
 * WHAT THIS IS NOT. It is not a fly that understands grand strategy. It is a
 * real neural simulation wired to a game, and it plays roughly as well as that
 * sounds. The point is that every spike is the model's, not a heuristic wearing
 * a fly costume.
 *
 * THE NUMBERS THAT SHAPED IT, all measured rather than assumed:
 *   - the connectome is 91 MB and a mod gets 64 MiB. Pruning to the neurons on
 *     a sensory->descending path keeps 97.7% of them, so there is nothing to
 *     prune. Delta+varint encoding is lossless and gets it to 42.8 MB.
 *   - one decision is ~19.1M neuron integrations, past the 100M per-turn fuel
 *     ceiling, which is why it declares fuelPerDecision.
 *   - one decision costs ~0.8 s, which is why it plays ONE country. Deciding
 *     for every AI country would put minutes on every turn.
 */
#include "gearbox.h"
#include "fly_brain.h"

#define ASSET "brain.odfly"

/* WHERE MEMORY COMES FROM, behind two functions rather than inline, so this
 * same file compiles for wasm AND for tools/decide_test.c. A decision path that
 * could only run inside the game would be a decision path nobody had run. */
#if defined(__wasm__)
extern unsigned char __heap_base;
static uint8_t *heap_base(void) { return (uint8_t *)&__heap_base; }
static uint64_t heap_bytes(void) {
    return (uint64_t)__builtin_wasm_memory_size(0) * 65536ull
         - (uint64_t)(uintptr_t)&__heap_base;
}
static int heap_reserve(uint64_t need) {
    if (heap_bytes() >= need) return 1;
    uint64_t base = (uint64_t)(uintptr_t)&__heap_base;
    uint64_t want_pages = (base + need + 65535ull) / 65536ull;
    uint64_t have_pages = (uint64_t)__builtin_wasm_memory_size(0);
    if (want_pages <= have_pages) return 1;
    return __builtin_wasm_memory_grow(0, (uint32_t)(want_pages - have_pages)) != (uint32_t)-1;
}
#else
/* The native test build supplies its own block; see tools/decide_test.c. */
extern uint8_t *fly_test_heap;
extern uint64_t fly_test_heap_bytes;
static uint8_t *heap_base(void) { return fly_test_heap; }
static uint64_t heap_bytes(void) { return fly_test_heap_bytes; }
static int heap_reserve(uint64_t need) { return fly_test_heap_bytes >= need; }
#endif

static FlyArena  g_arena;
static FlyBrain  g_brain;
static int       g_ready;

/* THE ONE SEAT. Pinned to the first AI country the host asks about, and never
 * moved: a fly that answered for everybody would put ~0.8 s on every AI
 * country's every decision, which is minutes a turn on a full map. */
static uint32_t  g_seat = GEARBOX_INVALID;

/* One window per country-turn, not per module. The host asks once per decision
 * module -- economy, politics, war, navy -- and running the brain four times
 * for one position would cost four times as much and answer the same question.
 * Open Fly does the same: one window, then every group read off it. */
static int32_t   g_window_turn = -1;
static uint16_t *g_counts;

/* WHAT A DESCENDING NEURON'S FIRING MEANS, and why the raw count is the wrong
 * number to read.
 *
 * These neurons are not equally excitable. A handful fire hard in EVERY window
 * whatever the fly saw, so whichever action group they are dealt into wins
 * every turn -- measured here on fifteen positions with the senses swinging
 * from 0 to 200 Hz: the answer was action 0 on all fifteen, and group 0's mean
 * was highest every single time.
 *
 * So the decoder reads how much each neuron fired RELATIVE TO ITS OWN RECENT
 * BASELINE. A neuron that always fires says nothing by firing again; one that
 * fires more than it usually does is the signal. That is also how sensory
 * systems tend to work -- adaptation, not absolute rate.
 *
 * The baseline is an exponential moving average updated once per window, and
 * the deviation is measured BEFORE the update, so the number scored is
 * "compared to what we expected", not "compared to including this". */
static double  *g_dn_base;
static double  *g_dn_dev;
static int      g_base_warm;
#define BASE_ALPHA 0.25

/* The encoder's memory of last turn, for "what CHANGED" -- land won or lost and
 * wars newly declared are signals; absolute size is not. */
static int       g_have_prev;
static double    g_prev_share;
static uint8_t  *g_prev_war;          /* one byte per country */
static uint32_t  g_war_cap;

/* Open Fly's reference scales, copied from web/decide.js REFS rather than
 * chosen here -- the same position must map to the same rates in both, and an
 * earlier version of this file invented these numbers while claiming it had
 * not. They are: land_pp 0.5, reserve_turns 12, wars 3, 200 Hz, floor 10. */
#define REF_LAND_PP       0.5
#define REF_RESERVE_TURNS 12.0
#define REF_WARS          3.0
#define MAX_HZ            200.0
#define FLOOR_HZ          10.0
#define WINDOW_MS         200.0

/* How much thinking one decision may do, in neuron integrations.
 *
 * MEASURED AGAINST THE REAL RUNTIME, not reasoned about. Through WAMR a quiet
 * window costs ~700M-1,000M instructions and a busy one passes 2,000,000,000 --
 * which no Gearbox mod can be granted, because the interpreter's limit is an
 * int. The unit is integrations plus synapses decoded, and the value is
 * whatever measurement showed keeps the WORST turn inside the budget -- the
 * first guess of 30M did not, because it counted only integrations. */
#define MAX_INTEGRATIONS  12000000ull
#define BRAIN_SEED        20240922u

static void logline(const char *s) {
    uint32_t n = 0; while (s[n]) n++;
    gearbox_log(GEARBOX_LOG_INFO, s, n);
}

#if !defined(__wasm__)
/* Test-only: tools/decide_test.c reads the senses directly rather than
 * inferring them from the choice they led to. Absent from the wasm build. */
double fly_debug_rates[FLY_CHANNELS];
double fly_debug_dn_total;
double fly_debug_score[16];
double fly_debug_size[16];
#define FLY_DEBUG_RATES(r) do { for (int _i = 0; _i < FLY_CHANNELS; _i++) \
                                    fly_debug_rates[_i] = (r)[_i]; } while (0)
#else
#define FLY_DEBUG_RATES(r) ((void)0)
#endif

static double clamp01(double x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }
static double dmax(double a, double b) { return a > b ? a : b; }

/* ------------------------------------------------------------- mod_load ---- */

int32_t mod_load(void) {
    uint32_t size = gearbox_asset_size(ASSET, sizeof(ASSET) - 1);
    if (size == 0) { logline("fly: brain.odfly is missing from the package"); return 1; }

    /* Reserve it all in one go. The brain is 42.8 MB and the simulator needs
     * ~5 MB beside it; growing incrementally would only add ways to fail
     * halfway with the asset already read. */
    uint64_t need = (uint64_t)size + 8ull * 1024 * 1024;
    if (!heap_reserve(need)) {
        logline("fly: could not get memory for the connectome");
        return 1;
    }
    g_arena.base = heap_base();
    g_arena.used = 0;
    g_arena.cap  = heap_bytes();

    uint8_t *file = (uint8_t *)fly_alloc(&g_arena, size, 8);
    if (!file) { logline("fly: no room for the connectome"); return 1; }
    if (gearbox_asset_read(ASSET, sizeof(ASSET) - 1, file, size) != size) {
        logline("fly: short read of brain.odfly");
        return 1;
    }

    int rc = fly_open(&g_brain, file, size, &g_arena);
    if (rc != 0) {
        logline(rc == -1 ? "fly: brain.odfly has the wrong header"
              : rc == -2 ? "fly: brain.odfly is truncated"
                         : "fly: not enough memory for the simulator");
        return 1;
    }

    g_counts = (uint16_t *)fly_alloc(&g_arena, (uint64_t)g_brain.n * 2, 2);
    g_dn_base = (double *)fly_alloc(&g_arena, (uint64_t)g_brain.dn_count * 8, 8);
    g_dn_dev  = (double *)fly_alloc(&g_arena, (uint64_t)g_brain.dn_count * 8, 8);
    if (!g_counts || !g_dn_base || !g_dn_dev) {
        logline("fly: no room for the spike counts"); return 1;
    }
    for (uint32_t j = 0; j < g_brain.dn_count; j++) { g_dn_base[j] = 0.0; g_dn_dev[j] = 0.0; }

    g_ready = 1;
    logline("fly: 138,639 neurons awake; it will play the first AI country it is asked about");
    return 0;
}

void mod_unload(void) { g_ready = 0; }

/* ------------------------------------------------------- senses from a map -- */

static void sense(uint32_t cid, double rates[FLY_CHANNELS]) {
    const double owned = (double)gearbox_country_province_count(cid);
    const double world = (double)gearbox_province_count();
    const double share = world > 0.5 ? 100.0 * owned / world : 0.0;

    const double gross = gearbox_country_income_gross(cid);
    const double net   = gearbox_country_income_net(cid);
    const double tres  = gearbox_country_treasury(cid);
    const double ratio = gross > 1e-9 ? net / gross : 0.0;

    /* Wars now, and which of them are new since last turn. */
    uint32_t n = gearbox_country_count();
    if (n > g_war_cap) {
        uint8_t *fresh = (uint8_t *)fly_alloc(&g_arena, n, 1);
        if (fresh) {
            for (uint32_t i = 0; i < n; i++) fresh[i] = 0;
            g_prev_war = fresh; g_war_cap = n;
        } else {
            n = g_war_cap;                 /* out of arena: keep the old bound */
        }
    }
    double wars = 0, new_wars = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t other = gearbox_country_at(i);
        if (other == GEARBOX_INVALID || other == cid) continue;
        uint32_t at = gearbox_at_war(cid, other) ? 1u : 0u;
        if (at) {
            wars += 1.0;
            if (g_have_prev && g_prev_war && !g_prev_war[i]) new_wars += 1.0;
        }
        if (g_prev_war && i < g_war_cap) g_prev_war[i] = (uint8_t)at;
    }

    const double d_land = g_have_prev ? share - g_prev_share : 0.0;
    g_prev_share = share;
    g_have_prev = 1;

    /* web/decide.js, Encoder.signals. Gains and losses are separate channels
     * rather than one signed one, because a fly has no signed senses. */
    double sig[FLY_CHANNELS];
    sig[0] = dmax(0, d_land) / REF_LAND_PP + dmax(0, ratio);              /* sugar  */
    sig[1] = dmax(0, -d_land) / REF_LAND_PP + dmax(0, -ratio)
           + 0.5 * new_wars;                                              /* bitter */
    sig[2] = gross > 1e-9 ? tres / (REF_RESERVE_TURNS * gross) : 0.0;     /* water  */
    sig[3] = wars / REF_WARS;                                             /* jon    */

    for (int c = 0; c < FLY_CHANNELS; c++) {
        double r = MAX_HZ * clamp01(sig[c]);
        /* Below the floor is silence, not a whisper: a rate of 0.3 Hz over a
         * 200 ms window is one spike in fifteen windows, which is noise
         * pretending to be a signal. */
        rates[c] = r >= FLOOR_HZ ? r : 0.0;
    }
    FLY_DEBUG_RATES(rates);
}

/* --------------------------------------------------------- the decision ---- */

/* Which action group a descending neuron belongs to, for this module.
 *
 * Dealt from a seeded generator rather than chosen: nobody knows which fly
 * neuron should mean "raise taxes", and pretending to would be the dishonest
 * part of this mod. The deal is DETERMINISTIC -- same seed, same module, same
 * action count, same partition every time -- so a given fly plays the same way
 * on the same position, and a different action count reshuffles honestly
 * rather than silently reusing a mapping built for a different decision.
 */
static uint32_t group_of(uint32_t dn_index, uint32_t module, uint32_t actions) {
    uint32_t s = g_brain.partition_seed ^ (module * 0x9E3779B9u) ^ (dn_index * 0x85EBCA6Bu);
    return fly_rand_u32(&s) % actions;
}

uint32_t mod_ai_choose(uint32_t country, uint32_t module) {
    if (!g_ready) return GEARBOX_INVALID;

    /* Pin the seat on the first AI country we are asked about. */
    if (g_seat == GEARBOX_INVALID) {
        if (!gearbox_country_is_ai(country)) return GEARBOX_INVALID;
        g_seat = country;
        logline("fly: taking this country");
    }
    if (country != g_seat) return GEARBOX_INVALID;

    uint32_t actions = gearbox_action_count(module);
    if (actions == 0) return GEARBOX_INVALID;

    /* The mask decides what may be answered at all. Asking for it first means a
     * turn with one legal move costs no brain time. */
    static char legal[256];
    uint32_t n_legal = gearbox_action_valid(module, legal, sizeof legal);
    if (n_legal == 0) return GEARBOX_INVALID;
    if (n_legal > sizeof legal) n_legal = sizeof legal;
    if (n_legal > actions) n_legal = actions;

    uint32_t only = GEARBOX_INVALID, count = 0;
    for (uint32_t a = 0; a < n_legal; a++) if (legal[a]) { only = a; count++; }
    if (count == 0) return GEARBOX_INVALID;
    if (count == 1) return only;      /* no choice: do not wake the fly */

    /* One window per country-turn, shared by every module. */
    int32_t turn = (int32_t)gearbox_turn_number();
    if (turn != g_window_turn) {
        double rates[FLY_CHANNELS];
        sense(g_seat, rates);
        /* Seeded by the turn, so the same position on the same turn gives the
         * same window -- a replay of a game plays out the same way. */
        uint32_t seed = BRAIN_SEED ^ ((uint32_t)turn * 2654435761u);
        fly_run_window(&g_brain, rates, WINDOW_MS, seed, g_counts, MAX_INTEGRATIONS);
        g_window_turn = turn;

        /* Deviation first, then update: the number scored is how this window
         * compared with what the last few led us to expect. */
        for (uint32_t j = 0; j < g_brain.dn_count; j++) {
            double c = (double)g_counts[g_brain.dn[j]];
            g_dn_dev[j] = g_base_warm ? c - g_dn_base[j] : 0.0;
            g_dn_base[j] = g_base_warm
                ? g_dn_base[j] + BASE_ALPHA * (c - g_dn_base[j])
                : c;
        }
        /* The first window of a game establishes the baseline and decides
         * nothing -- there is nothing yet for it to be a deviation FROM, and
         * ParrotZero plays that turn. */
        g_base_warm = 1;
    }

    /* Score each legal action by the MEAN spikes of its group, not the sum.
     *
     * decode.choose does it this way (`s / g.length`) and the reason shows up
     * the moment the groups are uneven: a sum rewards whichever group happens
     * to have been dealt more neurons, so the same group wins every window
     * whatever the fly saw. Measured here before the fix -- the answer was
     * action 0 on all fifteen positions tested. */
    static double score[256];
    static double size[256];
    uint32_t cap_a = actions < 256 ? actions : 256;
    for (uint32_t a = 0; a < cap_a; a++) { score[a] = 0.0; size[a] = 0.0; }

    for (uint32_t j = 0; j < g_brain.dn_count; j++) {
        uint32_t grp = group_of(j, module, actions);
        if (grp >= cap_a) continue;
        size[grp] += 1.0;
        score[grp] += g_dn_dev[j];
    }
    for (uint32_t a = 0; a < cap_a; a++)
        if (size[a] > 0.0) score[a] /= size[a];
#if !defined(__wasm__)
    fly_debug_dn_total = 0;
    for (uint32_t j = 0; j < g_brain.dn_count; j++)
        fly_debug_dn_total += g_counts[g_brain.dn[j]];
    for (uint32_t a = 0; a < 16 && a < cap_a; a++) {
        fly_debug_score[a] = score[a]; fly_debug_size[a] = size[a];
    }
#endif

    /* Ties broken by a draw seeded on the turn, as decode.choose does. Without
     * it every tie resolves to the lowest index, which on a quiet turn is not a
     * decision so much as a habit. */
    uint32_t tie_seed = BRAIN_SEED ^ ((uint32_t)turn * 40503u) ^ (module * 2246822519u);
    double best_score = -1.0, best_tie = 0.0;
    uint32_t best = GEARBOX_INVALID;
    for (uint32_t a = 0; a < n_legal; a++) {
        if (!legal[a]) continue;
        double t = (double)fly_rand_u32(&tie_seed) / 4294967296.0;
        if (score[a] > best_score || (score[a] == best_score && t < best_tie)) {
            best_score = score[a]; best_tie = t; best = a;
        }
    }

    /* Silence is an answer, but not this one. If no descending neuron in any
     * legal group fired, the fly has no opinion -- and saying so lets
     * ParrotZero decide rather than making the fly's indifference look like a
     * preference for whichever action happens to sort first. */
    if (best == GEARBOX_INVALID || best_score <= 0.0) return GEARBOX_INVALID;
    return best;
}
